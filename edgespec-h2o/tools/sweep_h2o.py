#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H2O parameter sweep & baseline comparison harness (P2-4 / P2-5 / P2-6).

Compares 4 baselines:
1. Full: full KV cache (unbounded / context size)
2. Local: recent tokens only (n_hh=0, n_pin=0, n_local=budget)
3. H2O (lambda=1.0): standard H2O without decay
4. H2O (lambda<1.0): time-decayed H2O (default lambda=0.98)

Sweeps across:
- Budgets: 10%, 20%, 30%, 50%, 100%
- Decay lambda: 1.0, 0.99, 0.98, 0.95
- Pins: 0, 1, 4

Usage:
    python tools/sweep_h2o.py --quick
    python tools/sweep_h2o.py --model-size 125m --device cuda
"""

from __future__ import print_function

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgespec.config import OPT_CONFIGS, RESULTS_DIR
from edgespec.model.opt import OPTModel


def create_sample_text_tokens(vocab_size=50272, seq_len=512, seed=42):
    """Generate or load real text token sequences for evaluation."""
    # Attempt to load real token IDs from golden data
    golden_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'golden', 'opt-125m', 'golden_token_ids.json'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'golden', 'golden_token_ids.json'),
    ]
    for gp in golden_paths:
        if os.path.exists(gp):
            try:
                with open(gp, 'r') as f:
                    data = json.load(f)
                all_ids = []
                for item in data:
                    tids = item.get('token_ids', [])
                    if tids:
                        all_ids.extend(tids)
                if len(all_ids) >= seq_len:
                    return torch.tensor([all_ids[:seq_len]], dtype=torch.long)
                elif len(all_ids) > 0:
                    repeats = (seq_len // len(all_ids)) + 1
                    tiled = (all_ids * repeats)[:seq_len]
                    return torch.tensor([tiled], dtype=torch.long)
            except Exception:
                pass

    # Fallback to deterministic structured tokens
    torch.manual_seed(seed)
    common_tokens = torch.randint(10, 100, (seq_len // 4,), dtype=torch.long)
    other_tokens = torch.randint(100, min(vocab_size, 10000), (seq_len - len(common_tokens),), dtype=torch.long)
    tokens = torch.cat([common_tokens, other_tokens])
    perm = torch.randperm(seq_len)
    tokens = tokens[perm]
    return tokens.unsqueeze(0)


def evaluate_window_loss(model, input_ids):
    """Evaluate cross-entropy loss over an input sequence using prefill_h2o."""
    with torch.no_grad():
        logits = model.prefill_h2o(input_ids)
        shift_logits = logits[0, :-1, :].float()
        shift_labels = input_ids[0, 1:]
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
        return loss.item()


def run_baseline_comparison(model, input_ids, max_positions=2048, device='cpu'):
    """Run the 4 standard baselines (P2-4) at 20% budget."""
    seq_len = input_ids.shape[1]
    budget_tokens = int(seq_len * 0.20)
    pin = 4
    local = min(128, budget_tokens // 2)
    hh = max(0, budget_tokens - local - pin)

    baselines = {}

    # 1. Full cache baseline
    model.init_h2o_caches(
        n_hh=seq_len, n_local=seq_len, n_pin=0, n_scratch=8,
        decay=1.0, device=device
    )
    t0 = time.time()
    full_loss = evaluate_window_loss(model, input_ids)
    full_time = time.time() - t0
    full_ppl = math.exp(full_loss)
    full_mem = model.total_h2o_memory_bytes() / (1024.0 * 1024.0)
    baselines['Full'] = {
        'loss': full_loss, 'ppl': full_ppl,
        'time_sec': full_time, 'mem_mib': full_mem,
        'delta_ppl': 0.0,
    }

    # 2. Local only (recent tokens only, no heavy hitters, no pins)
    model.init_h2o_caches(
        n_hh=0, n_local=budget_tokens, n_pin=0, n_scratch=8,
        decay=1.0, device=device
    )
    t0 = time.time()
    local_loss = evaluate_window_loss(model, input_ids)
    local_time = time.time() - t0
    local_ppl = math.exp(local_loss)
    local_mem = model.total_h2o_memory_bytes() / (1024.0 * 1024.0)
    baselines['Local'] = {
        'loss': local_loss, 'ppl': local_ppl,
        'time_sec': local_time, 'mem_mib': local_mem,
        'delta_ppl': local_ppl - full_ppl,
    }

    # 3. H2O (lambda=1.0, standard)
    model.init_h2o_caches(
        n_hh=hh, n_local=local, n_pin=pin, n_scratch=8,
        decay=1.0, device=device
    )
    t0 = time.time()
    h2o_std_loss = evaluate_window_loss(model, input_ids)
    h2o_std_time = time.time() - t0
    h2o_std_ppl = math.exp(h2o_std_loss)
    h2o_std_mem = model.total_h2o_memory_bytes() / (1024.0 * 1024.0)
    baselines['H2O (lambda=1.0)'] = {
        'loss': h2o_std_loss, 'ppl': h2o_std_ppl,
        'time_sec': h2o_std_time, 'mem_mib': h2o_std_mem,
        'delta_ppl': h2o_std_ppl - full_ppl,
    }

    # 4. H2O (lambda=0.98, time-decayed)
    model.init_h2o_caches(
        n_hh=hh, n_local=local, n_pin=pin, n_scratch=8,
        decay=0.98, device=device
    )
    t0 = time.time()
    h2o_dec_loss = evaluate_window_loss(model, input_ids)
    h2o_dec_time = time.time() - t0
    h2o_dec_ppl = math.exp(h2o_dec_loss)
    h2o_dec_mem = model.total_h2o_memory_bytes() / (1024.0 * 1024.0)
    baselines['H2O (lambda=0.98)'] = {
        'loss': h2o_dec_loss, 'ppl': h2o_dec_ppl,
        'time_sec': h2o_dec_time, 'mem_mib': h2o_dec_mem,
        'delta_ppl': h2o_dec_ppl - full_ppl,
    }

    return baselines


def run_full_sweep(model, input_ids, quick=False, device='cpu'):
    """Run parameter sweep over budgets, lambdas, and pins."""
    seq_len = input_ids.shape[1]
    budgets = [10, 20, 30] if quick else [10, 20, 30, 50]
    lambdas = [1.0, 0.98] if quick else [1.0, 0.99, 0.98, 0.95]
    pins = [1, 4] if quick else [0, 1, 4]

    sweep_records = []

    for b in budgets:
        budget_tokens = max(16, int(seq_len * b / 100))
        for p in pins:
            if p >= budget_tokens:
                continue
            local = min(128, max(8, (budget_tokens - p) // 2))
            hh = max(0, budget_tokens - p - local)

            for lam in lambdas:
                model.init_h2o_caches(
                    n_hh=hh, n_local=local, n_pin=p, n_scratch=8,
                    decay=lam, device=device
                )
                t0 = time.time()
                loss = evaluate_window_loss(model, input_ids)
                eval_time = time.time() - t0
                ppl = math.exp(loss)
                mem = model.total_h2o_memory_bytes() / (1024.0 * 1024.0)

                record = {
                    'budget_pct': b,
                    'budget_tokens': budget_tokens,
                    'n_pin': p,
                    'n_local': local,
                    'n_hh': hh,
                    'decay': lam,
                    'loss': round(loss, 4),
                    'ppl': round(ppl, 2),
                    'mem_mib': round(mem, 2),
                    'time_sec': round(eval_time, 3),
                }
                sweep_records.append(record)

    return sweep_records


def main():
    parser = argparse.ArgumentParser(description='H2O parameter sweep & baselines')
    parser.add_argument('--model-size', type=str, default='125m', choices=['125m', '1.3b'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--quick', action='store_true', default=False)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  H2O Cache Sweep & Baseline Harness (P2-4 / P2-5)")
    print("  Model: opt-%s  |  Device: %s  |  Context: %d tokens" %
          (args.model_size, args.device, args.seq_len))
    print("=" * 70)

    config = OPT_CONFIGS['opt-%s' % args.model_size]
    model = OPTModel(config, dtype=torch.float32 if args.device == 'cpu' else torch.float16)
    model.lm_head_weight = model.embed_tokens.weight
    model = model.to(args.device)
    model.eval()

    input_ids = create_sample_text_tokens(
        vocab_size=config['vocab_size'],
        seq_len=args.seq_len,
    ).to(args.device)

    # 1. Baseline comparisons (P2-4)
    print("\n--- Running P2-4 Baselines (at 20% budget) ---")
    baselines = run_baseline_comparison(model, input_ids, config.get('max_positions', 2048), args.device)

    print("\n%-20s %-10s %-10s %-12s %-12s" %
          ("Baseline", "Loss", "PPL", "Delta PPL", "Memory (MiB)"))
    print("-" * 65)
    for name, data in baselines.items():
        print("%-20s %-10.4f %-10.2f %-+12.2f %-12.2f" %
              (name, data['loss'], data['ppl'], data['delta_ppl'], data['mem_mib']))
    print("-" * 65)

    # 2. Parameter sweep (P2-5)
    print("\n--- Running P2-5 Parameter Sweep ---")
    sweep = run_full_sweep(model, input_ids, quick=args.quick, device=args.device)

    print("\n%-10s %-8s %-8s %-8s %-10s %-10s %-10s" %
          ("Budget %", "Pin", "Local", "HH", "Decay", "PPL", "Mem (MiB)"))
    print("-" * 70)
    for r in sweep:
        print("%-10d %-8d %-8d %-8d %-10.2f %-10.2f %-10.2f" %
              (r['budget_pct'], r['n_pin'], r['n_local'], r['n_hh'],
               r['decay'], r['ppl'], r['mem_mib']))
    print("-" * 70)

    # Recalibration Checkpoint (P2-6)
    # Find best configuration at 20% budget
    b20_records = [r for r in sweep if r['budget_pct'] == 20]
    best_b20 = min(b20_records, key=lambda x: x['loss']) if b20_records else None

    summary = {
        'model': 'opt-%s' % args.model_size,
        'seq_len': args.seq_len,
        'baselines': baselines,
        'sweep': sweep,
        'recalibration_p2_6': {
            'recommended_budget_pct': 20,
            'best_b20_config': best_b20,
        }
    }

    if best_b20:
        print("\nP2-6 Recalibration Summary:")
        print("  Best 20%% budget config: pin=%d, local=%d, hh=%d, decay=%.2f -> PPL=%.2f" %
              (best_b20['n_pin'], best_b20['n_local'], best_b20['n_hh'],
               best_b20['decay'], best_b20['ppl']))

    out_path = args.output or os.path.join(RESULTS_DIR, 'h2o_sweep_%s.json' % args.model_size)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSweep results saved to %s" % out_path)


if __name__ == '__main__':
    main()
