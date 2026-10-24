#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H2O parameter sweep & baseline comparison harness (P2-4 / P2-5 / P2-6).

Compares baselines at 20% budget:
1. Full: full KV cache (unbounded / context size)
2. Local: recent tokens only (n_hh=0, n_pin=0, n_local=budget)
3. H2O: across decay lambdas (1.0, 0.99, 0.98, 0.95) and pins (1, 4)

Evaluation methodology:
- Prefills the first N tokens (N = 25% of the window)
- Compresses the cache to budget
- Teacher-forces remaining tokens one at a time through H2O decode path
  (commit, update_scores, evict_to_budget)
- Computes loss ONLY on decoded tokens.
- Requires REAL weights (fails closed if missing).
- Enforces sanity check: loss < 8.0 and |local - full| > 0.01.

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


def find_candidate_weights(weights_path=None, model_size='125m'):
    """Return an existing path to real weights if available, else None."""
    candidates = []
    if weights_path:
        candidates.append(weights_path)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_w = os.environ.get('WEIGHTS_PATH')
    if env_w:
        candidates.append(env_w)

    candidates.extend([
        os.path.join(repo_root, 'data', 'weights', 'opt-%s-4bit.edgespec' % model_size),
        os.path.join(repo_root, 'data', 'weights', 'opt-%s.edgespec' % model_size),
        '/home/aiml/edgespec/edgespec-h2o/data/weights/opt-%s-4bit.edgespec' % model_size,
        '/home/aiml/edgespec/edgespec-h2o/data/weights/opt-%s.edgespec' % model_size,
        '/home/aiml/edgespec/models/opt-%s' % model_size,
        os.path.join(repo_root, 'models', 'opt-%s' % model_size),
    ])

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def load_real_model(weights_path=None, model_size='125m', device='cpu'):
    """Load model with real weights. Fails closed if missing."""
    p = find_candidate_weights(weights_path, model_size)
    dtype = torch.float32 if device == 'cpu' else torch.float16

    if p:
        if p.endswith('.edgespec'):
            from edgespec.model.loader import load_model
            return load_model(p, device=device, dtype=dtype, quantized=True)
        elif os.path.isdir(p) or p.endswith('.bin'):
            from edgespec.model.loader import load_model_from_hf
            return load_model_from_hf(p, device=device, dtype=dtype)

    # Fallback to HuggingFace hub download if transformers is installed
    try:
        from edgespec.model.loader import load_model_from_hf
        return load_model_from_hf('facebook/opt-%s' % model_size, device=device, dtype=dtype)
    except Exception as e:
        raise RuntimeError(
            "Real weights required for G2.5 quality evaluation, but none found. "
            "Checked local paths and HF fallback failed: %s" % str(e)
        )


def create_sample_text_tokens(vocab_size=50272, seq_len=512, seed=42):
    """Generate or load real text token sequences for evaluation."""
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

    # Deterministic fallback
    torch.manual_seed(seed)
    common_tokens = torch.randint(10, 100, (seq_len // 4,), dtype=torch.long)
    other_tokens = torch.randint(100, min(vocab_size, 10000), (seq_len - len(common_tokens),), dtype=torch.long)
    tokens = torch.cat([common_tokens, other_tokens])
    perm = torch.randperm(seq_len)
    tokens = tokens[perm]
    return tokens.unsqueeze(0)


def evaluate_window_loss(model, input_ids, prefill_fraction=0.25):
    """Evaluate cross-entropy loss by prefilling first N tokens (25% of window),
    compressing cache to budget, then teacher-forcing remaining tokens one at a time
    through H2O decode path (commit, update_scores, evict_to_budget).

    Computes loss ONLY on the decoded tokens.
    """
    seq_len = input_ids.shape[1]
    n_prefill = max(1, int(seq_len * prefill_fraction))
    prompt_ids = input_ids[:, :n_prefill]

    with torch.no_grad():
        # 1. Prefill first N tokens and compress cache to budget
        model.prefill_h2o(prompt_ids)
        for cache in model.h2o_caches:
            cache.evict_to_budget()

        # 2. Teacher-force remaining tokens one at a time through H2O decode path
        losses = []
        for t in range(n_prefill, seq_len - 1):
            curr_token = input_ids[:, t:t + 1]
            target_token = input_ids[0, t + 1:t + 2]

            for cache in model.h2o_caches:
                cache.begin_block(1)

            logits, all_probs = model.forward_block_h2o(curr_token, return_attn_probs=True)

            step_loss = F.cross_entropy(logits[0, 0:1, :].float(), target_token)
            losses.append(step_loss.item())

            for i, cache in enumerate(model.h2o_caches):
                cache.commit(1)
                if all_probs is not None and all_probs[i] is not None:
                    cache.update_scores(all_probs[i], 1)
                cache.evict_to_budget()

        if not losses:
            return 0.0
        return float(sum(losses) / len(losses))


def run_baseline_comparison(model, input_ids, max_positions=2048, device='cpu'):
    """Run Full, Local, and H2O baselines across requested lambdas and pins at 20% budget."""
    seq_len = input_ids.shape[1]
    budget_tokens = max(16, int(seq_len * 0.20))

    baselines = {}

    # 1. Full cache baseline
    print("  [1/10] Evaluating Full cache baseline...", flush=True)
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
    print("         -> loss=%.4f, ppl=%.2f (%.1fs)" % (full_loss, full_ppl, full_time), flush=True)

    # 2. Local only (recent tokens only, no heavy hitters, no pins)
    print("  [2/10] Evaluating Local cache baseline...", flush=True)
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
    print("         -> loss=%.4f, ppl=%.2f (%.1fs)" % (local_loss, local_ppl, local_time), flush=True)

    # Sanity assertion per prompt:
    # "loss must be < 8.0 on real weights, and Local at a 20% budget must differ from Full by > 0.01."
    if full_loss >= 8.0:
        raise AssertionError("Sanity check failed: Full loss %.4f >= 8.0 (real weights required)" % full_loss)

    local_diff = abs(local_loss - full_loss)
    if local_diff <= 0.01:
        raise AssertionError(
            "Sanity check failed: Local at 20%% budget (loss %.4f) differs from Full (loss %.4f) "
            "by %.5f <= 0.01" % (local_loss, full_loss, local_diff)
        )

    # 3. Report ppl for Full, Local, H2O at lambda 1.0, 0.99, 0.98, 0.95, and pin 1 and 4
    lambdas = [1.0, 0.99, 0.98, 0.95]
    pins = [1, 4]

    step_idx = 2
    for p in pins:
        local = min(128, max(8, (budget_tokens - p) // 2))
        hh = max(0, budget_tokens - p - local)

        for lam in lambdas:
            step_idx += 1
            print("  [%d/10] Evaluating H2O (pin=%d, lambda=%.2f)..." % (step_idx, p, lam), flush=True)
            model.init_h2o_caches(
                n_hh=hh, n_local=local, n_pin=p, n_scratch=8,
                decay=lam, device=device
            )
            t0 = time.time()
            loss = evaluate_window_loss(model, input_ids)
            eval_time = time.time() - t0
            ppl = math.exp(loss)
            mem = model.total_h2o_memory_bytes() / (1024.0 * 1024.0)

            key = "H2O (pin=%d, lambda=%.2f)" % (p, lam)
            baselines[key] = {
                'loss': loss, 'ppl': ppl,
                'time_sec': eval_time, 'mem_mib': mem,
                'delta_ppl': ppl - full_ppl,
                'pin': p, 'decay': lam,
            }
            print("         -> loss=%.4f, ppl=%.2f (%.1fs)" % (loss, ppl, eval_time), flush=True)

    # Also keep backward-compatible keys
    if 'H2O (pin=4, lambda=1.00)' in baselines:
        baselines['H2O (lambda=1.0)'] = baselines['H2O (pin=4, lambda=1.00)']
    if 'H2O (pin=4, lambda=0.98)' in baselines:
        baselines['H2O (lambda=0.98)'] = baselines['H2O (pin=4, lambda=0.98)']

    return baselines


def run_full_sweep(model, input_ids, quick=False, device='cpu', baselines=None):
    """Run parameter sweep over budgets, lambdas, and pins."""
    seq_len = input_ids.shape[1]
    budgets = [20] if quick else [10, 20, 30, 50]
    lambdas = [1.0, 0.98] if quick else [1.0, 0.99, 0.98, 0.95]
    pins = [1, 4]

    sweep_records = []

    for b in budgets:
        budget_tokens = max(16, int(seq_len * b / 100))
        for p in pins:
            if p >= budget_tokens:
                continue
            local = min(128, max(8, (budget_tokens - p) // 2))
            hh = max(0, budget_tokens - p - local)

            for lam in lambdas:
                key = "H2O (pin=%d, lambda=%.2f)" % (p, lam)
                if b == 20 and baselines and key in baselines:
                    b_entry = baselines[key]
                    loss = b_entry['loss']
                    ppl = b_entry['ppl']
                    mem = b_entry['mem_mib']
                    eval_time = b_entry['time_sec']
                else:
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
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--quick', action='store_true', default=False)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    # Enforce device sanity: do not claim cuda unless available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    print("\n" + "=" * 70)
    print("  H2O Cache Sweep & Baseline Harness (P2-4 / P2-5 / G2.5)")
    print("  Model: opt-%s  |  Device: %s  |  Context: %d tokens" %
          (args.model_size, args.device, args.seq_len))
    print("=" * 70)

    # Require real weights
    model = load_real_model(args.weights, model_size=args.model_size, device=args.device)

    config = OPT_CONFIGS['opt-%s' % args.model_size]
    input_ids = create_sample_text_tokens(
        vocab_size=config['vocab_size'],
        seq_len=args.seq_len,
    ).to(args.device)

    # 1. Baseline comparisons (P2-4 / G2.5)
    print("\n--- Running P2-4 Baselines (at 20% budget) ---")
    baselines = run_baseline_comparison(model, input_ids, config.get('max_positions', 2048), args.device)

    print("\n%-30s %-10s %-10s %-12s %-12s" %
          ("Baseline", "Loss", "PPL", "Delta PPL", "Memory (MiB)"))
    print("-" * 75)
    for name, data in baselines.items():
        if not name.startswith('H2O ('):
            print("%-30s %-10.4f %-10.2f %-+12.2f %-12.2f" %
                  (name, data['loss'], data['ppl'], data['delta_ppl'], data['mem_mib']))
    for name, data in baselines.items():
        if name.startswith('H2O ('):
            print("%-30s %-10.4f %-10.2f %-+12.2f %-12.2f" %
                  (name, data['loss'], data['ppl'], data['delta_ppl'], data['mem_mib']))
    print("-" * 75)

    # 2. Parameter sweep (P2-5)
    print("\n--- Running P2-5 Parameter Sweep ---")
    sys.stdout.flush()
    sweep = run_full_sweep(model, input_ids, quick=args.quick, device=args.device, baselines=baselines)

    print("\n%-10s %-8s %-8s %-8s %-10s %-10s %-10s" %
          ("Budget %", "Pin", "Local", "HH", "Decay", "PPL", "Mem (MiB)"))
    print("-" * 70)
    for r in sweep:
        print("%-10d %-8d %-8d %-8d %-10.2f %-10.2f %-10.2f" %
              (r['budget_pct'], r['n_pin'], r['n_local'], r['n_hh'],
               r['decay'], r['ppl'], r['mem_mib']))
    print("-" * 70)

    # Recalibration Checkpoint (P2-6)
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
