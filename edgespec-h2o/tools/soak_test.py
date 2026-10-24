#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Continuous generation soak test (P2-8 / G2.8).

Runs continuous generation with H2O cache on real model weights to verify
memory stability under sustained operation.

Pass criterion (Gate G2.8):
- 10-minute (600s) continuous generation: RSS drift <= 20 MiB,
  no OOM, temps logged.

Usage:
    python tools/soak_test.py --duration 600 --device cuda
    python tools/soak_test.py --duration 15 --device cpu  # fast test
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgespec.config import OPT_CONFIGS, RESULTS_DIR
from tools.sweep_h2o import load_real_model, create_sample_text_tokens


def get_current_rss_mib():
    """Get current process RSS in MiB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)
    except ImportError:
        # Fallback for Linux
        try:
            with open('/proc/self/statm', 'r') as f:
                pages = int(f.read().split()[1])
            return pages * os.sysconf('SC_PAGE_SIZE') / (1024.0 * 1024.0)
        except Exception:
            return 0.0


def read_system_temps():
    """Read temperatures if available (Jetson thermal zones or psutil)."""
    temps = {}
    for i in range(8):
        zone_path = '/sys/devices/virtual/thermal/thermal_zone%d/temp' % i
        type_path = '/sys/devices/virtual/thermal/thermal_zone%d/type' % i
        if os.path.exists(zone_path) and os.path.exists(type_path):
            try:
                with open(type_path, 'r') as f:
                    name = f.read().strip()
                with open(zone_path, 'r') as f:
                    val = float(f.read().strip()) / 1000.0
                temps[name] = val
            except Exception:
                pass

    if not temps:
        try:
            import psutil
            if hasattr(psutil, "sensors_temperatures"):
                st = psutil.sensors_temperatures()
                for k, v in st.items():
                    if v:
                        temps[k] = v[0].current
        except Exception:
            pass

    return temps


def run_soak_test(duration_sec=600, model_size='125m', device='cpu', weights_path=None):
    """Run continuous decode loop for duration_sec on real model and track RSS."""
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'

    print("\n" + "=" * 65)
    print("  Soak Test (G2.8): opt-%s on %s for %d seconds" %
          (model_size, device, duration_sec))
    print("=" * 65)

    # Require real weights
    model = load_real_model(weights_path=weights_path, model_size=model_size, device=device)
    config = OPT_CONFIGS['opt-%s' % model_size]

    # Initialize H2O cache with 20% budget
    budget = int(config.get('max_positions', 2048) * 0.20)
    n_local = min(128, budget // 2)
    n_pin = 4
    n_hh = max(0, budget - n_local - n_pin)
    model.init_h2o_caches(
        n_hh=n_hh, n_local=n_local, n_pin=n_pin, n_scratch=8,
        decay=0.98, device=device,
    )

    # Short prefill using sample tokens
    prompt = create_sample_text_tokens(vocab_size=config['vocab_size'], seq_len=32).to(device)
    with torch.no_grad():
        logits = model.prefill_h2o(prompt)
    curr_token = logits[0, -1, :].argmax(dim=-1).item()

    # Warmup steps
    for _ in range(5):
        for cache in model.h2o_caches:
            cache.begin_block(1)
        inp = torch.tensor([[curr_token]], device=device, dtype=torch.long)
        with torch.no_grad():
            logits, all_probs = model.forward_block_h2o(inp)
        curr_token = logits[0, -1, :].argmax(dim=-1).item()
        for i, cache in enumerate(model.h2o_caches):
            cache.commit(1)
            if all_probs and all_probs[i] is not None:
                cache.update_scores(all_probs[i], 1)
            cache.evict_to_budget()

    baseline_rss = get_current_rss_mib()
    print("Baseline RSS after warmup: %.2f MiB" % baseline_rss)

    start_time = time.time()
    steps = 0
    rss_history = [baseline_rss]
    sample_interval = max(1.0, min(10.0, duration_sec / 10.0))
    last_sample_time = start_time

    while (time.time() - start_time) < duration_sec:
        for cache in model.h2o_caches:
            cache.begin_block(1)
        inp = torch.tensor([[curr_token]], device=device, dtype=torch.long)
        with torch.no_grad():
            logits, all_probs = model.forward_block_h2o(inp)
        curr_token = logits[0, -1, :].argmax(dim=-1).item()
        for i, cache in enumerate(model.h2o_caches):
            cache.commit(1)
            if all_probs and all_probs[i] is not None:
                cache.update_scores(all_probs[i], 1)
            cache.evict_to_budget()

        steps += 1
        now = time.time()
        if now - last_sample_time >= sample_interval:
            cur_rss = get_current_rss_mib()
            rss_history.append(cur_rss)
            elapsed = now - start_time
            print("  [t=%5.1fs] steps=%-5d  RSS=%.2f MiB (drift=%+.2f MiB)" %
                  (elapsed, steps, cur_rss, cur_rss - baseline_rss))
            last_sample_time = now

    final_rss = get_current_rss_mib()
    peak_rss = max(rss_history + [final_rss])
    rss_drift = final_rss - baseline_rss
    elapsed = time.time() - start_time
    tok_per_sec = steps / elapsed if elapsed > 0 else 0

    temps = read_system_temps()

    passed = rss_drift <= 20.0
    print("\n" + "-" * 65)
    print("Soak Test Summary:")
    print("  Duration        : %.1f s" % elapsed)
    print("  Total tokens    : %d" % steps)
    print("  Throughput      : %.2f tok/s" % tok_per_sec)
    print("  Baseline RSS    : %.2f MiB" % baseline_rss)
    print("  Final RSS       : %.2f MiB" % final_rss)
    print("  Peak RSS        : %.2f MiB" % peak_rss)
    print("  RSS Drift       : %+.2f MiB (Threshold: <= 20.0 MiB)" % rss_drift)
    print("  Status          : %s" % ("PASS" if passed else "FAIL"))
    if temps:
        print("  Temperatures    : %s" % temps)
    print("-" * 65)

    return {
        "duration_sec": elapsed,
        "total_tokens": steps,
        "tok_per_sec": tok_per_sec,
        "baseline_rss_mib": baseline_rss,
        "final_rss_mib": final_rss,
        "peak_rss_mib": peak_rss,
        "rss_drift_mib": rss_drift,
        "threshold_mib": 20.0,
        "passed": passed,
        "temps": temps,
    }


def main():
    parser = argparse.ArgumentParser(description='Continuous soak test')
    parser.add_argument('--duration', type=int, default=600,
                        help='Duration in seconds (default: 600 = 10 min)')
    parser.add_argument('--model-size', type=str, default='125m',
                        choices=['125m', '1.3b'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    res = run_soak_test(
        duration_sec=args.duration,
        model_size=args.model_size,
        device=args.device,
        weights_path=args.weights,
    )

    out_path = args.output or os.path.join(RESULTS_DIR, 'soak_test_%s.json' % args.model_size)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)
    print("Results saved to %s" % out_path)

    sys.exit(0 if res['passed'] else 1)


if __name__ == '__main__':
    main()
