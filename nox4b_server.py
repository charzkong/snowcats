"""Standalone FastAPI server for llm-semantic-router/Decision-1.0-Nox-4B on NVIDIA.

The official bundle is AMD-only because of one runtime guard
(code/profile_guard.py). It pins Triton launch configs for FLA's
l2norm_fwd_kernel that were tuned on AMD gfx942, and reads the ROCm-only
`gcnArchName` GPU attribute. The weights themselves are plain BF16/FP32
safetensors with nothing vendor-specific.

This server loads the bundle's own code/decision_api.py and replaces only
`prepare_runtime_profile` with a no-op, so FLA autotunes its kernels for the
local GPU instead. Backbone, head, prompt rendering and temperature
calibration are the bundle's own, unchanged. Outputs are NOT numerically
validated against the published benchmarks (different kernels/torch build).

Runs in the ms-swift venv with flash-linear-attention==0.5.2 installed --no-deps.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import importlib.util
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import snapshot_download

REPO = "llm-semantic-router/Decision-1.0-Nox-4B"
REVISION = "0bb833504965c0eabdb9630b7bbd385cb2fe5cd4"  # pinned commit, 2026-09-23

app = FastAPI(title="Decision-1.0-Nox-4B")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
runtime_info = {}


class QuestionIn(BaseModel):
    type: str  # "choice", "score", or "noul"
    text: str
    options: Optional[List[str]] = None  # choice: labels; score: levels low to high; noul: ignored


class PredictRequest(BaseModel):
    state: str
    questions: List[QuestionIn]


def _load_engine(path: Path):
    spec = importlib.util.spec_from_file_location("nox_decision_api", path / "code/decision_api.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    # Skip the AMD-only FLA launch-profile guard; see module docstring.
    api.prepare_runtime_profile = lambda checkpoint, device="cuda:0": None

    manifest = json.loads((path / "bundle-manifest.json").read_text())
    temperatures = json.loads((path / "temperature.json").read_text())["temperatures"]
    return api.DecisionEngine(
        path,
        path / "code",
        device="cuda:0",
        max_length=manifest["input_length_limit"],
        batch_size=manifest["production_batch_size"],
        temperatures=temperatures,
        model_name="Decision-1.0-Nox-4B",
    )


@app.on_event("startup")
def startup():
    global engine, runtime_info
    print("Downloading/locating Decision-1.0-Nox-4B weights...")
    path = Path(snapshot_download(
        REPO,
        revision=REVISION,
        ignore_patterns=["assets/*", "metrics/*", "*.pdf", "*.png", "*.svg"],
    ))
    print("Loading Decision-1.0-Nox-4B...")
    engine = _load_engine(path)

    import torch, triton, fla, transformers
    runtime_info = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "fla": fla.__version__,
        "transformers": transformers.__version__,
        "validated_runtime": False,
    }
    print("Decision-1.0-Nox-4B ready.", runtime_info)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None, "runtime": runtime_info}


def _to_nox_question(q: QuestionIn):
    if q.type == "choice":
        if not q.options or len(q.options) < 2:
            raise HTTPException(status_code=400, detail="choice questions require at least 2 'options'")
        return {"type": "choice", "instructions": q.text, "criteria": {o: None for o in q.options}}
    if q.type == "score":
        if not q.options or not 2 <= len(q.options) <= 10:
            raise HTTPException(status_code=400, detail="score questions require 2..10 'options' (low to high)")
        return {"type": "score", "instructions": q.text, "criteria": list(q.options)}
    if q.type == "noul":
        return {"type": "noul", "instructions": q.text}
    raise HTTPException(status_code=400, detail=f"unknown question type: {q.type}")


@app.post("/api/nox4b/predict")
def predict(req: PredictRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly")
    try:
        questions = {f"q{i}": _to_nox_question(q) for i, q in enumerate(req.questions)}
        result = engine.decide(req.state, questions)
        return {
            "answers": [result["answers"][name] for name in questions],
            "usage": result["usage"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
