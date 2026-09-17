#!/usr/bin/env python3
"""RLCD, step 1: build the two contrastive-prompt datasets. Generation itself
goes through ms-swift's own `swift infer` (already this project's batch
inference tool - see tools/score_*.py's docstrings), not a hand-rolled
transformers.generate() loop.

    python3 prepare_rlcd_prompts.py --data data/rl_prompts.jsonl

writes data/rlcd_positive.jsonl and data/rlcd_negative.jsonl, both in the
same {"messages": [...]} shape as data/sft.jsonl, so `swift infer` needs no
extra conversion:

    swift infer --model ~/models/Qwen3.5-4B --adapters outputs/<sft-run>/adapter \\
        --val_dataset data/rlcd_positive.jsonl --max_new_tokens 96 \\
        --result_path outputs/rlcd_positive_result.jsonl
    swift infer --model ~/models/Qwen3.5-4B --adapters outputs/<sft-run>/adapter \\
        --val_dataset data/rlcd_negative.jsonl --max_new_tokens 96 \\
        --result_path outputs/rlcd_negative_result.jsonl

Then merge_rlcd_results.py turns the two result files into preference pairs.

Positive/negative suffixes are appended to the same base instruction, sampling
the same underlying model two different ways on the same source sentence -
that contrast is the entire RLCD idea; nothing here needs a dedicated library.
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))

POSITIVE_SUFFIX = (
    '特に、小学生や日本語を勉強し始めた人にも分かるように、'
    'できるだけ簡単な言葉と少ない漢字を使ってください。'
)
NEGATIVE_SUFFIX = (
    '元の文で使われている難しい漢字や表現は、できるだけそのまま残してください。'
)


def contrastive_prompt(base_prompt, suffix):
    instruction, _, sentence = base_prompt.strip().rpartition('\n')
    return f'{instruction}{suffix}\n{sentence}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'rl_prompts.jsonl'))
    ap.add_argument('--out-positive', default=os.path.join(HERE, 'data', 'rlcd_positive.jsonl'))
    ap.add_argument('--out-negative', default=os.path.join(HERE, 'data', 'rlcd_negative.jsonl'))
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding='utf-8') if l.strip()]

    with open(a.out_positive, 'w', encoding='utf-8') as fp, \
         open(a.out_negative, 'w', encoding='utf-8') as fn:
        for r in rows:
            pos = contrastive_prompt(r['prompt'], POSITIVE_SUFFIX)
            neg = contrastive_prompt(r['prompt'], NEGATIVE_SUFFIX)
            fp.write(json.dumps({'messages': [{'role': 'user', 'content': pos}]},
                                ensure_ascii=False) + '\n')
            fn.write(json.dumps({'messages': [{'role': 'user', 'content': neg}]},
                                ensure_ascii=False) + '\n')

    print(f'wrote {a.out_positive} and {a.out_negative}: {len(rows)} prompts each')
    print('Run `swift infer` on each (see module docstring), then merge_rlcd_results.py')


if __name__ == '__main__':
    main()
