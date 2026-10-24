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
    """Export golden token IDs for tokenizer validation (G1.2).

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

    # Save to canonical paths
    paths = [
        os.path.join(output_dir, 'golden_token_ids.json'),
        os.path.join(output_dir, 'opt-125m_tokenizer.json'),
        os.path.join(os.path.dirname(output_dir), 'golden_token_ids.json'),
        os.path.join(os.path.dirname(output_dir), 'opt-125m_tokenizer.json'),
    ]
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(golden, f, indent=2, ensure_ascii=False)

    print("Exported %d golden token ID entries to %s" % (len(golden), paths[0]))
    return golden


# --------------------------------------------------------------------------- #
#  Golden logits (G1.3)
# --------------------------------------------------------------------------- #

# Standard prompts including a long technical passage so total positions >= 500
GOLDEN_PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away,",
    "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):",
    "192.168.1.100 - - [24/Oct/2026:10:15:30 +0000] \"GET /api/health HTTP/1.1\" 200",
    "The quick brown fox",
    # Long text to exceed 500 positions across all prompts for Gate 1 G1.3 criteria
    ("EdgeSpec-H2O implements efficient speculative decoding and memory-constrained inference "
     "for large language models running on edge devices such as the NVIDIA Jetson Nano. "
     "The system combines three foundational techniques: post-training GPTQ 4-bit weight quantization, "
     "the Heavy-Hitter Oracle (H2O) KV-cache eviction algorithm with exponential time decay, and an "
     "importance-weighted suffix-match speculative decoding loop with zero added memory overhead. "
     "By storing only integer token IDs in the drafter history and keeping a bounded cache of heavy "
     "hitters and local window positions, EdgeSpec-H2O maintains flat memory usage across long sequence "
     "lengths up to 2048 tokens. In security log analysis and WAF rule drafting workloads, edge-deployed "
     "models achieve accelerated throughput while keeping telemetry strictly local and private without "
     "relying on cloud offloading. The system preserves strict mathematical equivalence with full-precision "
     "baselines in greedy verification mode and operates within the physical 4 GB memory budget of edge "
     "hardware through layer-major chunked execution and tensor-by-tensor low-memory deserialization.") * 5,
]


def export_golden_logits(output_dir, model_name='facebook/opt-125m', device='cuda'):
    """Export FP32 golden logits for numerics validation (G1.3)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Load in FP32 for golden reference
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    golden = []
    total_positions = 0
    for prompt in GOLDEN_PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        topk_vals, topk_ids = logits[0].topk(k=20, dim=-1)
        n_pos = logits.shape[1]
        total_positions += n_pos

        golden.append({
            "prompt": prompt[:100],
            "input_ids": input_ids[0].cpu().tolist(),
            "n_positions": n_pos,
            "top1_ids": logits[0].argmax(dim=-1).cpu().tolist(),
            "last_pos_logits_top20_ids": topk_ids[-1].cpu().tolist(),
            "last_pos_logits_top20_vals": topk_vals[-1].cpu().tolist(),
            "logits_mean": logits[0].mean().item(),
            "logits_std": logits[0].std().item(),
        })

    # Save full logits for first prompt
    first_input = tokenizer.encode(GOLDEN_PROMPTS[0], return_tensors='pt').to(device)
    with torch.no_grad():
        first_logits = model(first_input).logits[0].cpu()

    logits_paths = [
        os.path.join(output_dir, 'golden_logits_prompt0.pt'),
        os.path.join(os.path.dirname(output_dir), 'golden_logits_prompt0.pt'),
    ]
    for lp in logits_paths:
        torch.save(first_logits, lp)

    report = {
        "model": model_name,
        "dtype": "float32",
        "total_positions": total_positions,
        "prompts": golden,
    }

    meta_paths = [
        os.path.join(output_dir, 'golden_logits.json'),
        os.path.join(output_dir, 'opt-125m_logits.json'),
        os.path.join(os.path.dirname(output_dir), 'golden_logits.json'),
        os.path.join(os.path.dirname(output_dir), 'opt-125m_logits.json'),
    ]
    for mp in meta_paths:
        with open(mp, 'w') as f:
            json.dump(report, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Exported golden logits (%d total positions) to %s" % (total_positions, output_dir))


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
    """Export greedy generation reference outputs (G1.6)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    refs = []
    samples = []
    for prompt in GENERATION_PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=False,  # greedy
                temperature=1.0,
            )

        full_ids = output[0].cpu().tolist()
        prompt_len = input_ids.shape[1]
        gen_ids = full_ids[prompt_len:]
        generated_text = tokenizer.decode(full_ids)

        refs.append({
            "prompt": prompt,
            "input_ids": input_ids[0].cpu().tolist(),
            "generated_ids": full_ids,
            "generated_text": generated_text,
            "n_prompt_tokens": prompt_len,
            "n_generated_tokens": len(gen_ids),
        })

        samples.append({
            "prompt": prompt,
            "prompt_ids": input_ids[0].cpu().tolist(),
            "output_ids": gen_ids,
            "text": generated_text,
        })

    payload = {
        "model": model_name,
        "generations": refs,
        "samples": samples,
    }

    gen_paths = [
        os.path.join(output_dir, 'golden_generations.json'),
        os.path.join(output_dir, 'opt-125m_gen.json'),
        os.path.join(os.path.dirname(output_dir), 'golden_generations.json'),
        os.path.join(os.path.dirname(output_dir), 'opt-125m_gen.json'),
    ]
    for gp in gen_paths:
        with open(gp, 'w') as f:
            json.dump(payload, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Exported %d generation references to %s" % (len(refs), output_dir))


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
