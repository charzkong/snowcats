"""Standalone FastAPI server for monotykamary/LFM2.5-2.6B-RLCD.

Runs using the ms-swift venv's existing torch/transformers/accelerate stack -
no separate environment needed for this model. Run from inside the
LFM2.5-2.6B-RLCD repo directory so its bundled `pcd` package and local model
files resolve correctly.
"""
import os
os.environ.setdefault("USE_TF", "0")

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pcd import Engine

app = FastAPI(title="LFM2.5-2.6B-RLCD")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[Engine] = None


@app.on_event("startup")
def startup():
    global engine
    print("Loading LFM2.5-2.6B-RLCD (local weights, ms-swift torch stack)...")
    engine = Engine(device="cuda", dtype="float16", attention="sdpa", model_id=".", local_files_only=True)
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
