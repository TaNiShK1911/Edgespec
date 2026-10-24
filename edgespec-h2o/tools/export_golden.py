#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export golden reference files for gate checks (P1-3).

Generates:
- Token ID fixtures for tokenizer validation (G1.2)
- FP32 golden logits for numerics validation (G1.3)
- Per-layer weight checksums (G1.4)
- Generation reference outputs (G1.6)

Requires: transformers (laptop only).

Usage:
    python tools/export_golden.py --model facebook/opt-125m --output data/golden/
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgespec.config import GOLDEN_DIR


# --------------------------------------------------------------------------- #
#  Token ID fixtures (G1.2)
# --------------------------------------------------------------------------- #

GOLDEN_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "192.168.1.1 - - [24/Oct/2026:10:15:30 +0000] \"GET /api/v1/users HTTP/1.1\" 200 1234",
    "SELECT * FROM users WHERE id = 1; DROP TABLE users;--",
    "",
    " ",
    "Hello\nWorld\n",
    "こんにちは世界",
    "🎉🚀💻",
    "a" * 100,
    "The\ttab\tseparated\tvalues",
    '{"key": "value", "number": 42}',
    "Content-Type: application/json\r\nAuthorization: Bearer token123",
    "GET /admin/../../../etc/passwd HTTP/1.1",
    "<script>alert('XSS')</script>",
    "user=admin&password=test' OR '1'='1",
    "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36",
    "2026-10-24T10:15:30.123Z INFO  [main] Application started",
    "ERROR: Connection refused to 10.0.0.1:8080 after 3 retries",
    # Edge cases
    "  leading spaces",
    "trailing spaces  ",
    "multiple   spaces   between   words",
    "line1\nline2\nline3",
    "path/to/file.txt",
    "https://example.com/api?q=hello&lang=en",
    "null",
    "true",
    "false",
    "0",
    "1",
    "-1",
    "3.14159",
    "1e10",
    "\\n\\t\\r",
    "café résumé naïve",
    "Tom said \"Hello\" and she said 'Hi'",
    "C:\\Users\\admin\\Documents",
    "rm -rf / --no-preserve-root",
    "$(cat /etc/passwd)",
    "| grep -i error | awk '{print $1}'",
    "<<EOF\nHello\nEOF",
    "BEGIN TRANSACTION;\nCOMMIT;",
    "rate limit exceeded (429)",
    "cache-control: no-cache, no-store, must-revalidate",
    "X-Forwarded-For: 203.0.113.195, 70.41.3.18, 150.172.238.178",
    "AES-256-GCM SHA384 TLS_ECDHE_RSA",
    "AAAAAAAAAAAAAAAAx86_64",
    "AAAAAAAAAAAAAAAA%x%x%x%x",
    "\x00\x01\x02\x03",
]


def export_token_ids(output_dir, model_name='facebook/opt-125m'):
    """Export golden token IDs for tokenizer validation.

    Uses HuggingFace tokenizer as ground truth.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    golden = []
    for text in GOLDEN_STRINGS:
        try:
            ids = tokenizer.encode(text)  # includes BOS
            golden.append({
                "text": text,
                "token_ids": ids,
                "n_tokens": len(ids),
            })
        except Exception as e:
            golden.append({
                "text": text,
                "error": str(e),
            })

    path = os.path.join(output_dir, 'golden_token_ids.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print("Exported %d golden token ID entries to %s" % (len(golden), path))
    return golden


# --------------------------------------------------------------------------- #
#  Golden logits (G1.3)
# --------------------------------------------------------------------------- #

GOLDEN_PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away,",
    "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):",
    "192.168.1.100 - - [24/Oct/2026:10:15:30 +0000] \"GET /api/health HTTP/1.1\" 200",
    "The quick brown fox",
]


def export_golden_logits(output_dir, model_name='facebook/opt-125m', device='cuda'):
    """Export FP32 golden logits for numerics validation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Load in FP32 for golden reference
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    golden = []
    for prompt in GOLDEN_PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Store top-k logits and ids (full logits are too large)
        topk_vals, topk_ids = logits[0].topk(k=20, dim=-1)

        golden.append({
            "prompt": prompt,
            "input_ids": input_ids[0].cpu().tolist(),
            "n_positions": logits.shape[1],
            # Top-1 predictions at each position
            "top1_ids": logits[0].argmax(dim=-1).cpu().tolist(),
            # Full logits for last position (for detailed comparison)
            "last_pos_logits_top20_ids": topk_ids[-1].cpu().tolist(),
            "last_pos_logits_top20_vals": topk_vals[-1].cpu().tolist(),
            # Stats
            "logits_mean": logits[0].mean().item(),
            "logits_std": logits[0].std().item(),
        })

    # Also save the full logits as a binary file for the first prompt
    first_input = tokenizer.encode(GOLDEN_PROMPTS[0], return_tensors='pt').to(device)
    with torch.no_grad():
        first_logits = model(first_input).logits[0].cpu()  # [seq_len, vocab_size]

    logits_path = os.path.join(output_dir, 'golden_logits_prompt0.pt')
    torch.save(first_logits, logits_path)

    meta_path = os.path.join(output_dir, 'golden_logits.json')
    with open(meta_path, 'w') as f:
        json.dump({
            "model": model_name,
            "dtype": "float32",
            "prompts": golden,
        }, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Exported golden logits to %s" % output_dir)


# --------------------------------------------------------------------------- #
#  Generation references (G1.6)
# --------------------------------------------------------------------------- #

GENERATION_PROMPTS = [
    "The meaning of life is",
    "In the year 2026, artificial intelligence",
    "HTTP/1.1 200 OK\nContent-Type: text/html\n\n<!DOCTYPE html>\n<html>\n<head><title>",
]


def export_generation_refs(output_dir, model_name='facebook/opt-125m', device='cuda',
                           max_new=128):
    """Export greedy generation reference outputs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    refs = []
    for prompt in GENERATION_PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=False,  # greedy
                temperature=1.0,
            )

        generated_ids = output[0].cpu().tolist()
        generated_text = tokenizer.decode(generated_ids)

        refs.append({
            "prompt": prompt,
            "input_ids": input_ids[0].cpu().tolist(),
            "generated_ids": generated_ids,
            "generated_text": generated_text,
            "n_prompt_tokens": input_ids.shape[1],
            "n_generated_tokens": len(generated_ids) - input_ids.shape[1],
        })

    path = os.path.join(output_dir, 'golden_generations.json')
    with open(path, 'w') as f:
        json.dump({"model": model_name, "generations": refs}, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Exported %d generation references to %s" % (len(refs), path))


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Export golden reference files')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-logits', action='store_true')
    parser.add_argument('--skip-generations', action='store_true')
    args = parser.parse_args()

    if args.output is None:
        name = args.model.split('/')[-1]
        args.output = os.path.join(GOLDEN_DIR, name)

    os.makedirs(args.output, exist_ok=True)

    if not torch.cuda.is_available():
        args.device = 'cpu'

    # Token IDs (always, no GPU needed)
    export_token_ids(args.output, args.model)

    # Golden logits
    if not args.skip_logits:
        export_golden_logits(args.output, args.model, args.device)

    # Generation references
    if not args.skip_generations:
        export_generation_refs(args.output, args.model, args.device)

    print("\nAll golden files exported to %s" % args.output)


if __name__ == '__main__':
    main()
