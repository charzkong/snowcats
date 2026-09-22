"""Standalone FastAPI server for the OpenJev/Verdict decision engine.

Run from inside the Verdict-open-jev repo directory, WITHOUT setting PYTHONPATH to
any other repo (the qwen/laya server's own `core` package has the same top-level
name as this package's internal `core` module and will shadow it if both are on
the path at once).
"""
import os
os.environ.setdefault("USE_TF", "0")

from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rlcd import DecisionEngine, Choice, Option

app = FastAPI(title="OpenJev/Verdict Decision Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None


class OptionIn(BaseModel):
    id: str
    description: str


class QueryIn(BaseModel):
    id: str
    question: str
    options: List[OptionIn]


class PredictRequest(BaseModel):
    context: str
    queries: List[QueryIn]


@app.on_event("startup")
def startup():
    global engine
    print("Loading OpenJev/Verdict decision engine...")
    engine = DecisionEngine()
    print("OpenJev engine ready.")


@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": engine is not None}


@app.post("/api/openjev/predict")
def predict(req: PredictRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine still loading, try again shortly")
    try:
        queries = [
            Choice(
                id=q.id,
                question=q.question,
                options=[Option(id=o.id, description=o.description) for o in q.options],
            )
            for q in req.queries
        ]
        result = engine.evaluate(context=req.context, queries=queries)
        return {
            "results": [
                {
                    "id": r.id,
                    "selected_id": r.selected_id,
                    "selected_probability": r.selected_probability,
                    "is_abstention": r.is_abstention,
                    "calibration_status": r.calibration_status,
                    "probabilities": r.probabilities,
                    "latency_ms": r.latency_ms,
                }
                for r in result.results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
