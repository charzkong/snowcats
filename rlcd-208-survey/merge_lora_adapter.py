#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model, producing a standalone HF
checkpoint directory. Needed because harshatheg/Qwen-2.5-1B-RLCD's torch
backend (core/engine_torch.py) loads MODEL_ID directly via
AutoModelForCausalLM.from_pretrained() - it has no --adapters-style option
of its own, unlike ms-swift. One-time step per checkpoint you want to test
through that engine.

    python3 merge_lora_adapter.py \\
        --base /home/ecs-user/models/Qwen3.5-2B \\
        --adapter outputs/ms-swift-grammar-judgment-sft/v0-.../checkpoint-XX \\
        --out outputs/grammar-judgment-sft-merged

Run this on the GPU box, in the ms-swift venv (torch/transformers/peft are
already installed there).
"""
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    if not os.path.isdir(a.adapter):
        raise SystemExit(f"adapter not found: {a.adapter}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(a.base)
    try:
        base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, a.adapter)
    model = model.merge_and_unload()

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f'merged checkpoint saved to {a.out}')


if __name__ == '__main__':
    main()
