"""Standalone FastAPI server for monotykamary/LFM2.5-2.6B-RLCD.

Runs using the ms-swift venv's existing torch/transformers/accelerate stack -
no separate environment needed for this model. Its requirements-pcd.txt pins
torch==2.14.0 (CUDA 13), which this box's driver can't run; the ms-swift
torch 2.7.1+cu126 works instead.

Downloads its own pinned snapshot into the default HF cache on startup (no
local_dir copy) and imports the `pcd` package bundled in that snapshot.
Shares the GPU with Lux-9B, so expandable_segments is set to reduce
fragmentation in the ~2 GB of headroom left.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download

REPO = "monotykamary/LFM2.5-2.6B-RLCD"
REVISION = "31455458983bdbdc41b69ebbcedabd0d5de299c9"  # pinned commit, 2026-09-24

app = FastAPI(title="LFM2.5-2.6B-RLCD")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None


@app.on_event("startup")
def startup():
    global engine
    print("Downloading/locating LFM2.5-2.6B-RLCD weights...")
    path = snapshot_download(REPO, revision=REVISION, ignore_patterns=["results/*"])
    sys.path.insert(0, path)
    from pcd import Engine

    print("Loading LFM2.5-2.6B-RLCD (ms-swift torch stack)...")
    engine = Engine(device="cuda", dtype="float16", attention="sdpa", model_id=path, local_files_only=True)
    print("LFM2.5-2.6B-RLCD ready.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/api/lfm25rlcd/predict")
def predict(req: Dict[str, Any]):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly")
    text = req.get("text")
    schema = req.get("schema")
    mode = req.get("mode", "token")
    if not text or not schema:
        raise HTTPException(status_code=400, detail="'text' and 'schema' are required")
    try:
        result = engine.constrained(text, schema, mode=mode)
        return {
            "object": result["object"],
            "elapsed_ms": result["elapsed_ms"],
            "calibrated": result["calibrated"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
