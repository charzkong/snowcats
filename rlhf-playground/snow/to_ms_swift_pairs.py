#!/usr/bin/env python3
"""Convert this folder's {prompt, chosen, rejected, metadata} pairs into
ms-swift's native RM/DPO schema, so `swift rlhf --rlhf_type rm` can train on
them directly - no hand-rolled trainer needed.

Same conversion `docs/ms-swift-run-plan.md` §2a already does for DPO,
generalized to any pairs file in this folder and extended to carry
`metadata.gap` through as ms-swift's optional `margin` column (RM-specific -
DPO's schema doesn't use it) when present, e.g. from build_rm_pairs.py's
score gap.

    python3 to_ms_swift_pairs.py data/dpo_pairs_ranked.jsonl data/dpo_pairs_ranked.msswift.jsonl

Then train directly with the framework's own CLI - see examples/train/rlhf/rm/train.sh
upstream, or README.md in this folder for the exact command used here:

    swift rlhf --rlhf_type rm --model ~/models/Qwen3.5-4B --tuner_type lora \\
        --dataset data/dpo_pairs_ranked.msswift.jsonl --torch_dtype bfloat16 \\
        --output_dir outputs/rm-ranked ...
"""
import json, sys


def convert(in_path, out_path):
    n = 0
    with open(in_path, encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as out:
        for line in f:
            r = json.loads(line)
            row = {
                'messages': [
                    {'role': 'user', 'content': r['prompt']},
                    {'role': 'assistant', 'content': r['chosen']},
                ],
                'rejected_response': r['rejected'],
            }
            gap = r.get('metadata', {}).get('gap')
            if gap is not None:
                row['margin'] = gap
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            n += 1
    print(f'wrote {out_path}: {n} rows in ms-swift RM/DPO schema')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(f'usage: python3 {sys.argv[0]} <in.jsonl> <out.jsonl>')
    convert(sys.argv[1], sys.argv[2])
