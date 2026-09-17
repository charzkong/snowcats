#!/usr/bin/env python3
"""RLCD, step 3: turn the two `swift infer` result files (positive-prompted,
negative-prompted) into preference pairs, matching rows by the shared source
sentence - the base prompt's LAST line is unchanged by either contrastive
suffix (see prepare_rlcd_prompts.py's contrastive_prompt()), so it's a safe
join key even if `swift infer` reorders its batch internally.

    python3 merge_rlcd_results.py \\
        --positive-result outputs/rlcd_positive_result.jsonl \\
        --negative-result outputs/rlcd_negative_result.jsonl \\
        --data data/rl_prompts.jsonl \\
        --out data/dpo_pairs_rlcd.jsonl

Output shape matches build_rm_pairs.py's (prompt/chosen/rejected/metadata),
so to_ms_swift_pairs.py consumes it unchanged.
"""
import argparse, json, os, re

THINK_BLOCK = re.compile(r'^\s*<think>.*?</think>\s*', re.DOTALL)


def strip_think(text):
    return THINK_BLOCK.sub('', text, count=1).strip()


def sentence_of(prompt_text):
    return prompt_text.strip().rpartition('\n')[-1].strip()


def load_by_sentence(result_path):
    by_sentence = {}
    with open(result_path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            prompt = row['messages'][0]['content']
            by_sentence[sentence_of(prompt)] = strip_think(row['response'])
    return by_sentence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--positive-result', required=True)
    ap.add_argument('--negative-result', required=True)
    ap.add_argument('--data', required=True, help='the original rl_prompts.jsonl, for base prompt + metadata')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    positive = load_by_sentence(a.positive_result)
    negative = load_by_sentence(a.negative_result)
    rows = [json.loads(l) for l in open(a.data, encoding='utf-8') if l.strip()]

    out_rows, skipped = [], []
    for r in rows:
        sentence = sentence_of(r['prompt'])
        chosen = positive.get(sentence)
        rejected = negative.get(sentence)
        if not chosen or not rejected:
            skipped.append({'sentence': sentence, 'reason': 'missing_from_a_result_file'})
            continue
        if chosen == rejected:
            skipped.append({'sentence': sentence, 'reason': 'no_contrast'})
            continue
        out_rows.append({
            'prompt': r['prompt'],
            'chosen': chosen,
            'rejected': rejected,
            'metadata': {**r.get('metadata', {}), 'pair_source': 'rlcd_contrastive'},
        })

    with open(a.out, 'w', encoding='utf-8') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'wrote {a.out}: {len(out_rows)}/{len(rows)} contrastive pairs, {len(skipped)} skipped')
    for s in skipped:
        print(f"  skipped: {s['reason']}  ({s['sentence'][:30]}...)")


if __name__ == '__main__':
    main()
