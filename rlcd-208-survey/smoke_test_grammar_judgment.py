#!/usr/bin/env python3
"""grammar_judgment smoke test - the real PoC for ticket #208, using an
ALREADY-TRAINED, ALREADY-GOOD checkpoint (92% SFT accuracy, see
docs/grammar-judgment-grpo-design.md) instead of the stock model used in
the first smoke test. This tests whether the engine's speed/format
benefits hold up on a model that's actually good at the task, and whether
its confidence is any better calibrated than the ~23-point overconfidence
gap found on jlpt_mcq's stock-model test.

Reuses, unmodified:
  - core.schema.StructuredSchema / core.engine - the same cloned
    harshatheg/Qwen-2.5-1B-RLCD repo code used in the other smoke tests
  - plugin/grammar_judgment_reward.py's grammar_judgment_reward() - this
    project's own real scorer
  - data/grammar_judgment/holdout.jsonl - this project's own real 100-row holdout

Needs the TORCH backend - MLX is hardcoded to a different model with no
override. On the GPU box (Linux/CUDA) torch is already the default
backend, no BACKEND override needed; MODEL_ID must point at a MERGED
checkpoint (see merge_lora_adapter.py - the torch backend has no
--adapters-style option of its own).

Usage (on the GPU box):
    MODEL_ID=outputs/grammar-judgment-sft-merged python3 smoke_test_grammar_judgment.py --n 100
"""
import argparse, json, os, random, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RLCD_REPO = os.path.join(HERE, 'Qwen-2.5-1B-RLCD')
HOLDOUT = os.path.join(REPO_ROOT, 'data', 'grammar_judgment', 'holdout.jsonl')

REFERENCE_ACCURACY = 0.92  # this project's own SFT checkpoint, same holdout, docs/grammar-judgment-grpo-design.md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100, help='number of holdout rows to test (max 100)')
    ap.add_argument('--holdout', default=HOLDOUT)
    ap.add_argument('--seed', type=int, default=20260827)
    a = ap.parse_args()

    if not os.path.isdir(RLCD_REPO):
        raise SystemExit(f"RLCD repo not found at {RLCD_REPO}\n"
                         f"Clone it first: git clone --depth 1 "
                         f"https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD {RLCD_REPO}")
    if not os.environ.get('MODEL_ID'):
        raise SystemExit("set MODEL_ID to your merged grammar_judgment checkpoint first "
                         "(see merge_lora_adapter.py) - e.g.:\n"
                         "  MODEL_ID=outputs/grammar-judgment-sft-merged python3 " + sys.argv[0])

    sys.path.insert(0, RLCD_REPO)
    sys.path.insert(0, os.path.join(REPO_ROOT, 'plugin'))
    from core.schema import StructuredSchema
    from core.engine import get_engine, run_parallel_generation, USE_MLX

    # plugin/grammar_judgment_reward.py unconditionally imports swift.rewards,
    # only present where ms-swift is installed - stub it, same as the jlpt_mcq
    # smoke test, rather than editing that file or installing ms-swift here.
    if 'swift.rewards' not in sys.modules:
        import types
        swift_stub = types.ModuleType('swift')
        rewards_stub = types.ModuleType('swift.rewards')
        rewards_stub.ORM = object
        rewards_stub.orms = {}
        swift_stub.rewards = rewards_stub
        sys.modules['swift'] = swift_stub
        sys.modules['swift.rewards'] = rewards_stub
    import grammar_judgment_reward as gj

    print(f"backend: {'MLX (Apple Silicon)' if USE_MLX else 'PyTorch/CUDA-CPU'}")
    if USE_MLX:
        print("WARNING: MLX backend ignores MODEL_ID - it always loads the hardcoded "
              "mlx-community/Qwen2.5-1.5B-Instruct-4bit, not your merged checkpoint. "
              "Set BACKEND=torch (and pip install torch transformers accelerate) to "
              "actually test your model.")

    tokenizer = get_engine()[1]  # (model, tokenizer) on MLX, (model, tokenizer, device) on torch

    schema = StructuredSchema(
        {'label': {'type': 'enum', 'choices': ['correct', 'incorrect'],
                   'description': 'whether the Japanese sentence is grammatically correct'}},
        tokenizer=tokenizer)

    rows = [json.loads(l) for l in open(a.holdout, encoding='utf-8') if l.strip()]
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.n]

    rewards, confidences, elapsed_ms_list, correct = [], [], [], 0
    for row in rows:
        result = run_parallel_generation(row['prompt'], schema)
        picked = result['parsed_json']['label']['value']
        confidence = result['parsed_json']['label']['prob']
        reward = gj.grammar_judgment_reward(picked, row['metadata']['label'])
        rewards.append(reward)
        confidences.append(confidence)
        elapsed_ms_list.append(result['elapsed_ms'])
        if reward == 1.0:
            correct += 1

    n = len(rows)
    acc = correct / n
    print(f'n={n}')
    print(f'accuracy: {correct}/{n} ({acc:.1%})')
    print(f'mean reward (grammar_judgment_reward scale): {statistics.mean(rewards):.3f}')
    print(f'mean confidence (RLCD field_telemetry): {statistics.mean(confidences):.3f}')
    print(f'mean elapsed_ms: {statistics.mean(elapsed_ms_list):.1f}')
    print()
    print(f'Reference - this project\'s own SFT checkpoint on this same holdout: '
          f'{REFERENCE_ACCURACY:.1%} (92/100), docs/grammar-judgment-grpo-design.md')
    delta = acc - REFERENCE_ACCURACY
    print(f'Delta vs. that reference: {delta:+.1%} '
          f'({"tool preserved the model\'s judgment" if abs(delta) <= 0.03 else "SOMETHING CHANGED - investigate before trusting this path"})')
    mean_conf = statistics.mean(confidences)
    print(f'\nConfidence-vs-accuracy gap: {mean_conf:.1%} reported vs. {acc:.1%} actual '
          f'({mean_conf - acc:+.1%}). For comparison, the stock model on jlpt_mcq showed '
          f'a +23.5pp overconfidence gap (62.7% reported vs 39.2% actual).')


if __name__ == '__main__':
    main()
