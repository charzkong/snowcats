"""Drop-in replacement for baseline_reward.py's `kanji_vocab_reward`, backed
by a reward model trained with `swift rlhf --rlhf_type rm` instead of a
hand-coded heuristic. Same ORM interface as plugin/plugin.py's
KanjiVocabReward, so it can be swapped into a real GRPO run by pointing
configs/rl.yaml's `grpo.reward_funcs` at `learned_reward` instead of
`kanji_vocab_reward` and adding this file to `grpo.external_plugins` - no
other training flag changes.

Scoring uses ms-swift's own inference engine (`swift.infer_engine`), the same
one shown in the framework's own `examples/train/rlhf/rm/infer.py` - not a
hand-rolled AutoModelForSequenceClassification forward pass. This keeps
scoring consistent with however the RM was actually trained/saved by the
`swift rlhf --rlhf_type rm` run.

Model path comes from an env var, not a constructor arg, because swift
instantiates ORM classes with no arguments:

    SNOW_RM_PATH=outputs/rm-ranked/checkpoint-XXX python3 -m swift rlhf ...

Falls back to outputs/rm-ranked if unset. See README.md for the full
train-then-plug-in sequence.
"""
import os
from typing import List

try:
    from swift.rewards import ORM, orms
except ImportError:  # only installed on the GPU training box
    ORM = object
    orms = {}

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
RM_PATH = os.environ.get('SNOW_RM_PATH', os.path.join(_REPO_ROOT, 'outputs', 'rm-ranked'))
BASE_MODEL = os.environ.get('SNOW_BASE_MODEL', os.path.expanduser('~/models/Qwen3.5-4B'))

_engine = None


def _load():
    global _engine
    if _engine is not None:
        return
    from swift.infer_engine import TransformersEngine
    if not os.path.isdir(RM_PATH):
        raise RuntimeError(
            f"reward model not found at {RM_PATH}\n"
            f"Train one first (see README.md):\n"
            f"  swift rlhf --rlhf_type rm --model {BASE_MODEL} --tuner_type lora \\\n"
            f"      --dataset data/dpo_pairs_ranked.msswift.jsonl --output_dir {RM_PATH} ...\n"
            f"or set SNOW_RM_PATH to point at an existing checkpoint.")
    _engine = TransformersEngine(
        BASE_MODEL, task_type='seq_cls', num_labels=1, problem_type='regression',
        adapters=[RM_PATH])


def learned_reward(source_text: str, generated: str) -> float:
    """source_text is `instruction\\nsentence`, matching every other reward
    in this project (see baseline_reward.py's _split_source)."""
    _load()
    from swift.infer_engine import InferRequest
    request = InferRequest(messages=[
        {'role': 'user', 'content': source_text},
        {'role': 'assistant', 'content': generated},
    ])
    response = _engine.infer([request])[0]
    return float(response.choices[0].message.content)


class LearnedReward(ORM):
    def __call__(self, completions, source_text, **kwargs) -> List[float]:
        rewards = []
        for completion, src in zip(completions, source_text):
            g = completion if isinstance(completion, str) else str(completion)
            rewards.append(learned_reward(src, g))
        return rewards


orms['learned_reward'] = LearnedReward
