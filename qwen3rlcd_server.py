"""Standalone FastAPI server for anthonym21/qwen3-0.6b-rlcd-decision.

Run from inside the eve-rlcd repo directory with `uv run uvicorn ...` so the
package's own `rlcd` module resolves via its own .venv - keep this as its own
process, since this package's top-level name (`rlcd`) collides with other
`rlcd`-named packages used by other models in this deployment.
"""
import os
os.environ.setdefault("USE_TF", "0")

from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rlcd.decide import ChoiceQ, ScoreQ, NoulQ, Decider

app = FastAPI(title="qwen3-0.6b-rlcd-decision")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

decider: Optional[Decider] = None
MODEL_DIR = "qwen3-rlcd-decision"


class QuestionIn(BaseModel):
    type: str  # "choice", "score", or "noul"
    text: str
    options: Optional[List[str]] = None  # required for choice/score, ignored for noul


class PredictRequest(BaseModel):
    state: str
    questions: List[QuestionIn]


@app.on_event("startup")
def startup():
    global decider
    print("Loading qwen3-0.6b-rlcd-decision...")
    decider = Decider.load(MODEL_DIR)
    print("qwen3-0.6b-rlcd-decision ready.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": decider is not None}


def _build_question(q: QuestionIn):
    if q.type == "choice":
        if not q.options:
            raise HTTPException(status_code=400, detail="choice questions require 'options'")
        return ChoiceQ(q.text, q.options)
    if q.type == "score":
        if not q.options:
            raise HTTPException(status_code=400, detail="score questions require 'options' (low to high)")
        return ScoreQ(q.text, q.options)
    if q.type == "noul":
        return NoulQ(q.text)
    raise HTTPException(status_code=400, detail=f"unknown question type: {q.type}")


@app.post("/api/qwen3rlcd/predict")
def predict(req: PredictRequest):
    if decider is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly")
    try:
        questions = [_build_question(q) for q in req.questions]
        answers = decider.ask(req.state, questions)
        return {"answers": answers}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
