"""Standalone FastAPI server for the Laya decision models.

Loads two checkpoints in one process and picks one per request:
- convaiinnovations/laya-typed-decisions (default): English specialist,
  ModernBERT-large, 421M params, fine-tuned on invoices / security incidents /
  customer service / agent traces.
- convaiinnovations/laya-multilingual (`"multilingual": true`): mmBERT-base,
  322M params, 100+ languages, not fine-tuned on those workflows.

Runs in the ms-swift venv with `pip install --no-deps laya==0.3.17` (laya is
pure Python; torch/transformers come from the venv). laya.Router is not used:
its auto-routing sends English text to a third checkpoint ("english") that we
don't load. Shares the GPU with Lux-9B and LFM2.5, so headroom is small.

laya silently falls back to CPU if the GPU runs out of memory; /health and
every response report the actual device.

Request format matches /api/nox4b and /api/lux9b. Choice questions may also
pass `descriptions` (same length as `options`) - these models were trained with
option descriptions and are more accurate with them.
"""
import os
os.environ.setdefault("USE_TF", "0")  # transformers' TF probe can deadlock laya.load()
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import snapshot_download

# name -> (repo, pinned commit, 2026-09-24)
CHECKPOINTS = {
    "typed-decisions": ("convaiinnovations/laya-typed-decisions", "1a793eb568e6718f15941d08f85432581df534e3"),
    "multilingual": ("convaiinnovations/laya-multilingual", "453577addb8a9c85395d2449a91fce4742f5f2f1"),
}
CHECKPOINT_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]

app = FastAPI(title="Laya decision models")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agents: Dict[str, Any] = {}


class QuestionIn(BaseModel):
    type: str  # "choice", "score", or "noul"
    text: str
    options: Optional[List[str]] = None  # choice: labels; score: levels low to high; noul: ignored
    descriptions: Optional[List[str]] = None  # choice only, optional: one per option


class PredictRequest(BaseModel):
    state: Any  # text, JSON object, or list of conversation turns
    questions: List[QuestionIn]
    multilingual: bool = False  # true: laya-multilingual; false: laya-typed-decisions (English)


@app.on_event("startup")
def startup():
    import laya
    for name, (repo, revision) in CHECKPOINTS.items():
        print(f"Downloading/locating {repo}...")
        path = snapshot_download(repo, revision=revision, allow_patterns=CHECKPOINT_FILES)
        print(f"Loading {repo}...")
        agents[name] = laya.load(path, device="cuda")
        print(f"{repo} ready on {agents[name].device}.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {name: str(agent.device) for name, agent in agents.items()},
    }


def _to_laya_question(q: QuestionIn):
    if q.type == "choice":
        if not q.options or len(q.options) < 2:
            raise HTTPException(status_code=400, detail="choice questions require at least 2 'options'")
        if len(set(q.options)) != len(q.options):
            raise HTTPException(status_code=400, detail="choice 'options' must be unique")
        if q.descriptions is not None and len(q.descriptions) != len(q.options):
            raise HTTPException(status_code=400, detail="'descriptions' must have one entry per option")
        descriptions = q.descriptions or q.options
        return {"type": "choice", "instructions": q.text, "criteria": dict(zip(q.options, descriptions))}
    if q.type == "score":
        if not q.options or not 2 <= len(q.options) <= 10:
            raise HTTPException(status_code=400, detail="score questions require 2..10 'options' (low to high)")
        return {"type": "score", "instructions": q.text, "criteria": list(q.options)}
    if q.type == "noul":
        return {"type": "noul", "instructions": q.text}
    raise HTTPException(status_code=400, detail=f"unknown question type: {q.type}")


@app.post("/api/laya/predict")
def predict(req: PredictRequest):
    name = "multilingual" if req.multilingual else "typed-decisions"
    agent = agents.get(name)
    if agent is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly")
    if not req.questions:
        raise HTTPException(status_code=400, detail="questions must be a nonempty list")
    try:
        questions = {f"q{i}": _to_laya_question(q) for i, q in enumerate(req.questions)}
        result = agent.predict(req.state, questions)
        return {
            "model": CHECKPOINTS[name][0],
            "answers": [result["answers"][qid] for qid in questions],
            "device": str(agent.device),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
