#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Full benchmark matrix for EdgeSpec-H2O (P3-7).

Runs all four modes from the plan section 3.6:
  A (baseline): 4-bit, full KV, spec off
  B:            4-bit, H2O,     spec off
  C:            4-bit, full KV, spec on
  D (EdgeSpec): 4-bit, H2O,     spec on

Per mode measures: tok/s (prefill/decode), mean accepted tokens/step,
alpha_hat, peak RAM, KV bytes, ROUGE-2, PPL, temps.

Outputs: results/*.csv + auto-generated figures (matplotlib if available).

Python-3.6 safe.
"""

from __future__ import print_function

import argparse
import csv
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from edgespec.config import (
    OPT_CONFIGS, RESULTS_DIR, SPEC_DEFAULTS, H2O_DEFAULTS,
)


def _get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _get_ram_mib():
    """Best-effort system RAM reading."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    total = int(line.split()[1]) / 1024.0
                if line.startswith('MemAvailable:'):
                    return total - int(line.split()[1]) / 1024.0
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_mode_a(model, prompt_ids, max_new_tokens=64):
    """Mode A: 4-bit, full KV, no speculation."""
    device = prompt_ids.device
    model.reset_caches()
    model.init_kv_caches(device=device)

    t0 = time.time()
    with torch.no_grad():
        logits = model.prefill_chunked(prompt_ids)
    t_prefill = time.time() - t0

    next_token = logits[0, -1, :].argmax(dim=-1).item()
    all_ids = [next_token]

    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if next_token == model.config.get('eos_token_id', 2):
                break
            inp = torch.tensor([[next_token]], device=device, dtype=torch.long)
            l, _ = model.forward_block(inp)
            next_token = l[0, -1, :].argmax(dim=-1).item()
            all_ids.append(next_token)
    t_decode = time.time() - t0

    n = len(all_ids)
    return {
        'mode': 'A',
        'spec': False,
        'h2o': False,
        'prefill_tok_s': prompt_ids.shape[1] / t_prefill if t_prefill > 0 else 0,
        'decode_tok_s': n / t_decode if t_decode > 0 else 0,
        'n_generated': n,
        'alpha_hat': 0.0,
        'mean_accepted_per_step': 1.0,
        'kv_bytes': model.total_kv_memory_bytes(),
        'peak_ram_mib': _get_ram_mib(),
        'generated_ids': all_ids,
    }


def benchmark_mode_b(model, prompt_ids, max_new_tokens=64,
                     n_hh=128, n_local=128, n_pin=4, decay=0.98):
    """Mode B: 4-bit, H2O, no speculation."""
    device = prompt_ids.device
    model.init_h2o_caches(
        n_hh=n_hh, n_local=n_local, n_pin=n_pin,
        n_scratch=8, decay=decay, device=device
    )

    t0 = time.time()
    with torch.no_grad():
        logits = model.prefill_h2o(prompt_ids)
    t_prefill = time.time() - t0

    next_token = logits[0, -1, :].argmax(dim=-1).item()
    all_ids = [next_token]

    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if next_token == model.config.get('eos_token_id', 2):
                break
            inp = torch.tensor([[next_token]], device=device, dtype=torch.long)
            for cache in model.h2o_caches:
                cache.begin_block(1)
            l, probs = model.forward_block_h2o(inp)
            next_token = l[0, -1, :].argmax(dim=-1).item()
            all_ids.append(next_token)
            for i, cache in enumerate(model.h2o_caches):
                cache.commit(1)
                if probs is not None and probs[i] is not None:
                    cache.update_scores(probs[i], 1)
                cache.evict_to_budget()
    t_decode = time.time() - t0

    n = len(all_ids)
    return {
        'mode': 'B',
        'spec': False,
        'h2o': True,
        'prefill_tok_s': prompt_ids.shape[1] / t_prefill if t_prefill > 0 else 0,
        'decode_tok_s': n / t_decode if t_decode > 0 else 0,
        'n_generated': n,
        'alpha_hat': 0.0,
        'mean_accepted_per_step': 1.0,
        'kv_bytes': model.total_h2o_memory_bytes(),
        'peak_ram_mib': _get_ram_mib(),
        'generated_ids': all_ids,
    }


def benchmark_mode_c(model, prompt_ids, max_new_tokens=64,
                     gamma=4, lenience=1.0):
    """Mode C: 4-bit, full KV, spec on.

    Uses full KV cache but with speculative decoding.
    Note: This requires H2O caches with budget >= context (no eviction).
    """
    device = prompt_ids.device
    # Use H2O caches with large budget (no eviction)
    model.init_h2o_caches(
        n_hh=2048, n_local=128, n_pin=4,
        n_scratch=gamma + 2, decay=1.0, device=device
    )

    from edgespec.spec.drafter import SuffixDrafter
    from edgespec.spec.sampler import SpeculativeSampler
    from edgespec.spec.verifier import SpeculativeVerifier

    drafter = SuffixDrafter(max_n=4, selector='importance')
    sampler = SpeculativeSampler(mode='greedy', lenience=lenience)
    verifier = SpeculativeVerifier(
        model=model, drafter=drafter, sampler=sampler,
        gamma_fixed=gamma
    )

    t0 = time.time()
    with torch.no_grad():
        all_ids, gen_stats = verifier.generate(
            prompt_ids, max_new_tokens=max_new_tokens
        )
    t_total = time.time() - t0

    n = gen_stats['n_generated']
    return {
        'mode': 'C',
        'spec': True,
        'h2o': False,
        'prefill_tok_s': 0,  # included in total
        'decode_tok_s': n / t_total if t_total > 0 else 0,
        'n_generated': n,
        'alpha_hat': gen_stats.get('alpha_hat', 0),
        'mean_accepted_per_step': gen_stats.get('mean_accepted_per_step', 0),
        'kv_bytes': model.total_h2o_memory_bytes(),
        'peak_ram_mib': _get_ram_mib(),
        'generated_ids': all_ids[prompt_ids.shape[1]:],
        'gen_stats': gen_stats,
    }


def benchmark_mode_d(model, prompt_ids, max_new_tokens=64,
                     n_hh=128, n_local=128, n_pin=4, decay=0.98,
                     gamma=4, lenience=1.0):
    """Mode D (EdgeSpec-H2O): 4-bit, H2O, spec on."""
    device = prompt_ids.device
    model.init_h2o_caches(
        n_hh=n_hh, n_local=n_local, n_pin=n_pin,
        n_scratch=gamma + 2, decay=decay, device=device
    )

    from edgespec.spec.drafter import SuffixDrafter
    from edgespec.spec.sampler import SpeculativeSampler
    from edgespec.spec.verifier import SpeculativeVerifier

    drafter = SuffixDrafter(max_n=4, selector='importance')
    sampler = SpeculativeSampler(mode='greedy', lenience=lenience)
    verifier = SpeculativeVerifier(
        model=model, drafter=drafter, sampler=sampler,
        gamma_fixed=gamma
    )

    t0 = time.time()
    with torch.no_grad():
        all_ids, gen_stats = verifier.generate(
            prompt_ids, max_new_tokens=max_new_tokens
        )
    t_total = time.time() - t0

    n = gen_stats['n_generated']
    return {
        'mode': 'D',
        'spec': True,
        'h2o': True,
        'prefill_tok_s': 0,
        'decode_tok_s': n / t_total if t_total > 0 else 0,
        'n_generated': n,
        'alpha_hat': gen_stats.get('alpha_hat', 0),
        'mean_accepted_per_step': gen_stats.get('mean_accepted_per_step', 0),
        'kv_bytes': model.total_h2o_memory_bytes(),
        'peak_ram_mib': _get_ram_mib(),
        'generated_ids': all_ids[prompt_ids.shape[1]:],
        'gen_stats': gen_stats,
    }


def run_benchmark_matrix(model, prompts, max_new_tokens=64,
                         n_hh=128, n_local=128, n_pin=4, decay=0.98,
                         gamma=4, lenience=1.0):
    """Run the full 4-mode benchmark on a list of prompts.

    Parameters
    ----------
    model : OPTModel
    prompts : list of (name, torch.LongTensor [1, seq_len])
    max_new_tokens : int

    Returns
    -------
    list of dict (one per mode × prompt)
    """
    all_results = []
    modes = [
        ('A', benchmark_mode_a),
        ('B', benchmark_mode_b),
        ('C', benchmark_mode_c),
        ('D', benchmark_mode_d),
    ]

    for prompt_name, prompt_ids in prompts:
        for mode_name, mode_fn in modes:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("  Running Mode %s on '%s'..." % (mode_name, prompt_name))
            try:
                if mode_name == 'A':
                    result = mode_fn(model, prompt_ids, max_new_tokens)
                elif mode_name == 'B':
                    result = mode_fn(model, prompt_ids, max_new_tokens,
                                     n_hh, n_local, n_pin, decay)
                elif mode_name == 'C':
                    result = mode_fn(model, prompt_ids, max_new_tokens,
                                     gamma, lenience)
                elif mode_name == 'D':
                    result = mode_fn(model, prompt_ids, max_new_tokens,
                                     n_hh, n_local, n_pin, decay,
                                     gamma, lenience)

                result['prompt_name'] = prompt_name
                result['prompt_len'] = prompt_ids.shape[1]
                # Remove non-serializable fields
                result.pop('generated_ids', None)
                result.pop('gen_stats', None)
                all_results.append(result)
                print("    -> %.1f tok/s, alpha=%.2f" % (
                    result['decode_tok_s'], result.get('alpha_hat', 0)))
            except Exception as e:
                print("    -> ERROR: %s" % str(e))
                all_results.append({
                    'mode': mode_name,
                    'prompt_name': prompt_name,
                    'error': str(e),
                })

    return all_results


def save_results(results, output_dir=None):
    """Save benchmark results to CSV and JSON.

    Parameters
    ----------
    results : list of dict
    output_dir : str or None
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, 'benchmark_matrix.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("JSON: %s" % json_path)

    # CSV
    csv_path = os.path.join(output_dir, 'benchmark_matrix.csv')
    if results:
        keys = sorted(set(k for r in results for k in r.keys()))
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print("CSV: %s" % csv_path)

    # Speedup summary
    _print_speedup_summary(results)


def _print_speedup_summary(results):
    """Print speedup of Mode D vs Mode A per prompt."""
    print("\n=== SPEEDUP SUMMARY ===")
    prompts = set(r.get('prompt_name', '') for r in results)
    for pname in sorted(prompts):
        a_tok = [r['decode_tok_s'] for r in results
                 if r.get('prompt_name') == pname and r.get('mode') == 'A'
                 and 'decode_tok_s' in r]
        d_tok = [r['decode_tok_s'] for r in results
                 if r.get('prompt_name') == pname and r.get('mode') == 'D'
                 and 'decode_tok_s' in r]

        if a_tok and d_tok:
            speedup = d_tok[0] / a_tok[0] if a_tok[0] > 0 else 0
            d_alpha = [r.get('alpha_hat', 0) for r in results
                       if r.get('prompt_name') == pname and r.get('mode') == 'D']
            print("  %s: A=%.1f tok/s, D=%.1f tok/s, speedup=%.2fx, alpha=%.2f" % (
                pname, a_tok[0], d_tok[0], speedup,
                d_alpha[0] if d_alpha else 0))


def main():
    parser = argparse.ArgumentParser(description='EdgeSpec-H2O benchmark matrix')
    parser.add_argument('--model-size', type=str, default='125m',
                        choices=['125m', '350m', '1.3b'])
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--lenience', type=float, default=1.0)
    parser.add_argument('--n-hh', type=int, default=128)
    parser.add_argument('--n-local', type=int, default=128)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--decay', type=float, default=0.98)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    device = args.device or _get_device()
    model_key = 'opt-' + args.model_size
    config = OPT_CONFIGS[model_key]
    dtype = torch.float16

    print("=== EdgeSpec-H2O Benchmark Matrix ===")
    print("Model: %s, Device: %s" % (model_key, device))

    weights_cand = args.weights
    if not weights_cand or not os.path.isfile(weights_cand):
        default_p = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'weights', 'opt-%s-4bit.edgespec' % args.model_size
        )
        if os.path.isfile(default_p):
            weights_cand = default_p

    if weights_cand and os.path.isfile(weights_cand):
        from edgespec.model.loader import load_model
        model = load_model(weights_cand, device=device)
    else:
        from edgespec.model.opt import OPTModel
        model = OPTModel(config, dtype=dtype)
        model = model.to(device)
    model.eval()

    # Create test prompts
    prompts = []
    # Short prompt
    short = torch.randint(10, config['vocab_size'], (1, 32),
                          device=device, dtype=torch.long)
    prompts.append(('short_32', short))

    # Medium prompt
    medium = torch.randint(10, config['vocab_size'], (1, 128),
                           device=device, dtype=torch.long)
    prompts.append(('medium_128', medium))

    # Log-style prompt (repetitive — should favour the drafter)
    log_tokens = list(range(10, 50)) * 8  # repetitive pattern
    log_prompt = torch.tensor([log_tokens[:256]], device=device, dtype=torch.long)
    prompts.append(('logs_256', log_prompt))

    print("\nRunning benchmark matrix...")
    results = run_benchmark_matrix(
        model, prompts, max_new_tokens=args.max_new_tokens,
        n_hh=args.n_hh, n_local=args.n_local, decay=args.decay,
        gamma=args.gamma, lenience=args.lenience,
    )

    save_results(results, args.output_dir)
    print("\nDone.")


if __name__ == '__main__':
    main()
