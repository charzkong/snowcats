"""
FastAPI Server for Parallel Constrained Decision Engine.
Serves interactive side-by-side benchmark UI, presets, live streaming endpoints,
the Laya multi-model classification router, and proxies to the standalone
OpenJev/Verdict decision engine (run separately to avoid a `core` package
name collision between this repo and the OpenJev package).
"""

import os
os.environ.setdefault("USE_TF", "0")  # avoid transformers/TF import deadlock

import json
import asyncio
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.schema import StructuredSchema
from core.engine import (
    get_engine,
    run_naive_generation,
    stream_naive_generation,
    run_parallel_generation,
    run_rlcd_generation,
)
from laya import Router as LayaRouter

app = FastAPI(title="Parallel Constrained Decision Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PRESETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets")
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")

OPENJEV_INTERNAL_URL = "http://localhost:18074/api/openjev/predict"

laya_router: Optional[LayaRouter] = None


class PredictRequest(BaseModel):
    context: str
    schema_def: Dict[str, Any] = Field(..., alias="schema")
    temperature: Optional[float] = None

    class Config:
        populate_by_name = True


class LayaPredictRequest(BaseModel):
    model: str
    state: Dict[str, Any]
    questions: Dict[str, Any]


@app.on_event("startup")
def on_startup():
    print("Pre-warming Qwen inference engine...")
    get_engine()
    print("Qwen engine ready.")

    global laya_router
    print("Preloading Laya models (multilingual, typed-decisions)...")
    laya_router = LayaRouter()
    laya_router.preload(["multilingual", "typed-decisions"])
    print("Laya models ready.")


@app.get("/api/presets")
def list_presets():
    presets = []
    if os.path.exists(PRESETS_DIR):
        for fname in sorted(os.listdir(PRESETS_DIR)):
            if fname.endswith(".json"):
                fpath = os.path.join(PRESETS_DIR, fname)
                try:
                    with open(fpath, "r") as f:
                        presets.append(json.load(f))
                except Exception as e:
                    print(f"Error loading preset {fname}: {e}")
    return presets


@app.post("/api/run-parallel")
@app.post("/api/run-rlcd")
def api_run_parallel(req: PredictRequest):
    try:
        schema = StructuredSchema(req.schema_def)
        temp = req.temperature if req.temperature is not None else 1.0
        res = run_parallel_generation(req.context, schema, temperature=temp)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/run-naive")
def api_run_naive(req: PredictRequest):
    try:
        schema = StructuredSchema(req.schema_def)
        temp = req.temperature if req.temperature is not None else 0.2
        res = run_naive_generation(req.context, schema, temperature=temp)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/stream-naive")
def api_stream_naive(req: PredictRequest):
    try:
        schema = StructuredSchema(req.schema_def)
        temp = req.temperature if req.temperature is not None else 0.2

        def event_generator():
            try:
                for event in stream_naive_generation(req.context, schema, temperature=temp):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                print(f"Error in stream_naive_generation: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/compare")
def api_compare(req: PredictRequest):
    try:
        schema = StructuredSchema(req.schema_def)
        naive_temp = req.temperature if req.temperature is not None else 0.2
        rlcd_temp = req.temperature if req.temperature is not None else 1.0

        naive_res = run_naive_generation(req.context, schema, temperature=naive_temp)
        rlcd_res = run_rlcd_generation(req.context, schema, temperature=rlcd_temp)

        speedup = naive_res["elapsed_ms"] / max(rlcd_res["elapsed_ms"], 1.0)
        steps_reduction = naive_res["sequential_forward_passes"] / max(rlcd_res["sequential_forward_passes"], 1.0)

        return {
            "speedup_multiplier": round(speedup, 1),
            "steps_reduction": round(steps_reduction, 1),
            "naive": naive_res,
            "parallel": rlcd_res,
            "rlcd": rlcd_res
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/laya/predict")
def api_laya_predict(req: LayaPredictRequest):
    if laya_router is None:
        raise HTTPException(status_code=503, detail="Laya models still loading, try again shortly")
    try:
        return laya_router.predict(req.state, req.questions, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/openjev/predict")
def api_openjev_predict(req: Dict[str, Any]):
    """Forwards to the standalone OpenJev/Verdict server running on :18074.
    Kept out-of-process deliberately: that package's own `core` module name
    collides with this repo's `core` package if imported into the same process."""
    body = json.dumps(req).encode()
    http_req = urllib.request.Request(
        OPENJEV_INTERNAL_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(http_req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=e.code, detail=e.read().decode())
    except urllib.error.URLError as e:
        raise HTTPException(status_code=503, detail=f"OpenJev backend unreachable: {e}")


# Mount web frontend
if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
