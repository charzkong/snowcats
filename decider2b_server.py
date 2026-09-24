"""Standalone FastAPI server for Mapika/decider-2b (v10).

Qwen3.5-2B-Base one-pass decision model. No AMD lock; runs in the ms-swift
venv as-is (torch 2.7.1, transformers with qwen3_5, flash-linear-attention
0.5.2 already installed for Nox/Lux). Imports the `decider` package bundled in
the downloaded snapshot.

CUDA graphs are off by default (DECIDER_GRAPHS=0): each captured input shape
keeps its own GPU memory, and this GPU is shared with other models. Set
DECIDER_GRAPHS=1 for much lower latency when there is room.

Request format matches /api/nox4b, /api/lux9b and /api/laya. Choice questions
may also pass `descriptions` (same length as `options`).
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import snapshot_download

REPO = "Mapika/decider-2b"
REVISION = "9839cc9d908be16c5988c0d041034b5fdf82c7a2"  # pinned commit (v10), 2026-09-24
USE_GRAPHS = os.environ.get("DECIDER_GRAPHS", "0") == "1"

app = FastAPI(title="decider-2b")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

decider = None


class QuestionIn(BaseModel):
    type: str  # "choice", "score", or "noul"
    text: str
    options: Optional[List[str]] = None  # choice: labels; score: levels low to high; noul: ignored
    descriptions: Optional[List[str]] = None  # choice only, optional: one per option


class PredictRequest(BaseModel):
    state: Any  # text, JSON object, or array
    questions: List[QuestionIn]


@app.on_event("startup")
def startup():
    global decider
    print("Downloading/locating decider-2b weights...")
    path = snapshot_download(REPO, revision=REVISION)
    sys.path.insert(0, path)
    from decider.infer import Decider

    print(f"Loading decider-2b (CUDA graphs {'on' if USE_GRAPHS else 'off'})...")
    decider = Decider(path, device="cuda", use_graphs=USE_GRAPHS)
    print(f"decider-2b ready ({decider.name}).")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": decider is not None,
        "version": decider.name if decider is not None else None,
        "cuda_graphs": USE_GRAPHS,
    }


def _to_decider_question(q: QuestionIn):
    if q.type == "choice":
        if not q.options or not 2 <= len(q.options) <= 255:
            raise HTTPException(status_code=400, detail="choice questions require 2..255 'options'")
        if len(set(q.options)) != len(q.options):
            raise HTTPException(status_code=400, detail="choice 'options' must be unique")
        if q.descriptions is not None and len(q.descriptions) != len(q.options):
            raise HTTPException(status_code=400, detail="'descriptions' must have one entry per option")
        descriptions = q.descriptions or [None] * len(q.options)
        return {"type": "choice", "instructions": q.text, "criteria": dict(zip(q.options, descriptions))}
    if q.type == "score":
        if not q.options or not 2 <= len(q.options) <= 10:
            raise HTTPException(status_code=400, detail="score questions require 2..10 'options' (low to high)")
        return {"type": "score", "instructions": q.text, "criteria": list(q.options)}
    if q.type == "noul":
        return {"type": "noul", "instructions": q.text}
    raise HTTPException(status_code=400, detail=f"unknown question type: {q.type}")


@app.post("/api/decider2b/predict")
def predict(req: PredictRequest):
    if decider is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly")
    if not req.questions:
        raise HTTPException(status_code=400, detail="questions must be a nonempty list")
    try:
        questions = {f"q{i}": _to_decider_question(q) for i, q in enumerate(req.questions)}
        result = decider.system_one(req.state, questions)
        return {
            "model": result.get("model"),
            "answers": [result["answers"][qid] for qid in questions],
            "usage": result.get("usage"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
