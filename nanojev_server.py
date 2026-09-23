"""Standalone FastAPI server for C-Tianyu/NanoJev.

Downloads only the files this server needs (weights, config, tokenizer,
the predictor script) - skips predictions/variants/stage1 files in the repo
to keep disk footprint small. Needs a CUDA 12.x torch build (the repo's own
requirements-toy.txt pins torch==2.14.0 / CUDA 13.x, which our driver does
not support - install a CUDA 12.x torch first, or reuse an existing venv
that already has one).
"""
import os
os.environ.setdefault("USE_TF", "0")

import sys
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download

app = FastAPI(title="NanoJev")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
predictor: Optional[Any] = None


@app.on_event("startup")
def startup():
    global predictor
    print("Downloading/locating NanoJev weights...")
    path = snapshot_download(
        "C-Tianyu/NanoJev",
        revision="unified-games-v1",
        local_dir=MODEL_DIR,
        allow_patterns=[
            "best.safetensors",
            "config.json",
            "backbone_config/*",
            "tokenizer/*",
            "source/scripts/*.py",
        ],
    )
    sys.path.insert(0, os.path.join(path, "source", "scripts"))
    from predict_toy_decisions import DecisionPredictor

    print("Loading NanoJev (cuda:0, bf16)...")
    predictor = DecisionPredictor(
        path, device_name="cuda:0", precision="bf16", disable_native_triton=False
    )
    print("NanoJev ready.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/api/nanojev/predict")
def predict(req: Dict[str, Any]):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly")
    try:
        return predictor.predict(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
