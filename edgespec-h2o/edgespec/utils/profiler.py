# -*- coding: utf-8 -*-
"""Profiling utilities for EdgeSpec-H2O.

Python-3.6 safe.  Measures:
- Prefill vs decode throughput (tok/s)
- Per-op GPU time vs CPU launch time
- RSS + tegrastats RAM
- Verify-cost probe V(k) for k = 1..8
"""

import gc
import os
import sys
import time
import json

import torch


class Timer(object):
    """Simple CPU wall-clock timer with CUDA sync."""

    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._start = None
        self._elapsed = 0.0

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self._start = time.time()
        return self

    def stop(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self._elapsed = time.time() - self._start
        return self._elapsed

    @property
    def elapsed(self):
        return self._elapsed


class CUDATimer(object):
    """GPU-side timer using CUDA events (more precise than wall clock)."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self._elapsed_ms = 0.0

    def start(self):
        self.start_event.record()
        return self

    def stop(self):
        self.end_event.record()
        torch.cuda.synchronize()
        self._elapsed_ms = self.start_event.elapsed_time(self.end_event)
        return self._elapsed_ms / 1000.0  # seconds

    @property
    def elapsed_ms(self):
        return self._elapsed_ms

    @property
    def elapsed(self):
        return self._elapsed_ms / 1000.0


def get_rss_mb():
    """Get current RSS (Resident Set Size) in MiB. Linux/macOS."""
    try:
        # Linux
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except (IOError, OSError):
        pass

    # macOS fallback
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_gpu_memory_mb():
    """Get GPU memory usage in MiB (allocated / reserved)."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved": torch.cuda.memory_reserved() / (1024 * 1024),
        "total": torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
            if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
            else 0.0,
    }


def get_system_memory():
    """Get system memory info from /proc/meminfo (Linux/Nano)."""
    info = {}
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split()
                if parts[0] in ('MemTotal:', 'MemFree:', 'MemAvailable:',
                                'Buffers:', 'Cached:', 'SwapTotal:',
                                'SwapFree:', 'Zram:'):
                    info[parts[0].rstrip(':')] = int(parts[1]) / 1024.0  # MiB
    except (IOError, OSError):
        pass
    return info


def get_swap_activity():
    """Check vmstat for swap in/out (si/so). Returns (si, so) or (0, 0)."""
    try:
        import subprocess
        result = subprocess.check_output(['vmstat', '1', '2'],
                                         stderr=subprocess.STDOUT)
        lines = result.decode().strip().split('\n')
        if len(lines) >= 3:
            # Last line has the 1-second sample
            vals = lines[-1].split()
            # vmstat columns vary; si is typically index 7, so index 8
            si = int(vals[7])
            so = int(vals[8])
            return si, so
    except Exception:
        pass
    return 0, 0


def benchmark_throughput(model, prompt_ids, max_new_tokens=32, warmup=3,
                         repeats=1, context_len=None):
    """Measure prefill and decode throughput.

    Parameters
    ----------
    model : OPTModel
    prompt_ids : torch.LongTensor [1, seq_len]
    max_new_tokens : int
    warmup : int
    repeats : int

    Returns
    -------
    dict with prefill_tok_per_sec, decode_tok_per_sec, etc.
    """
    device = prompt_ids.device
    prompt_len = prompt_ids.shape[1]

    # Warmup
    for _ in range(warmup):
        model.reset_caches()
        model.init_kv_caches(device=device)
        with torch.no_grad():
            model.generate_greedy(prompt_ids, max_new_tokens=2)

    results = []
    for _ in range(repeats):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model.reset_caches()
        model.init_kv_caches(device=device)

        # Prefill timing
        timer = Timer(use_cuda=True)
        timer.start()
        with torch.no_grad():
            logits = model.prefill_chunked(prompt_ids)
        prefill_time = timer.stop()

        next_token = logits[0, -1, :].argmax(dim=-1).item()

        # Decode timing
        timer.start()
        with torch.no_grad():
            generated_ids = [next_token]
            for step in range(max_new_tokens - 1):
                input_ids = torch.tensor([[generated_ids[-1]]], device=device,
                                         dtype=torch.long)
                step_logits, _ = model.forward_block(input_ids)
                nt = step_logits[0, -1, :].argmax(dim=-1).item()
                generated_ids.append(nt)
                if nt == model.config.get('eos_token_id', 2):
                    break
        decode_time = timer.stop()
        n_decoded = len(generated_ids)

        gpu_mem = get_gpu_memory_mb()

        results.append({
            "prompt_len": prompt_len,
            "n_decoded": n_decoded,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "prefill_tok_per_sec": prompt_len / prefill_time if prefill_time > 0 else 0,
            "decode_tok_per_sec": n_decoded / decode_time if decode_time > 0 else 0,
            "gpu_allocated_mb": gpu_mem["allocated"],
            "gpu_reserved_mb": gpu_mem["reserved"],
            "rss_mb": get_rss_mb(),
        })

    # Average
    avg = {}
    for key in results[0]:
        vals = [r[key] for r in results]
        avg[key] = sum(vals) / len(vals)
    avg["context_len"] = context_len or prompt_len

    return avg


def verify_cost_probe(model, context_len=512, k_range=None, warmup=3, repeats=5):
    """Measure the cost of a forward pass over k = 1..8 tokens at a given context.

    This is the critical Phase-1 measurement: V(k) determines whether
    speculative decoding is viable (β = (V(8)-V(1))/7).

    Parameters
    ----------
    model : OPTModel
    context_len : int
    k_range : list of int or None  (defaults to [1..8])
    warmup : int
    repeats : int

    Returns
    -------
    dict with V(k) values and computed β
    """
    if k_range is None:
        k_range = list(range(1, 9))

    device = next(model.parameters()).device

    # Create a dummy prompt to fill the KV cache to context_len
    dummy_prompt = torch.randint(
        10, model.config['vocab_size'], (1, context_len),
        device=device, dtype=torch.long
    )

    v_table = {}

    for k in k_range:
        times = []

        for trial in range(warmup + repeats):
            # Reset and fill cache to context_len
            model.reset_caches()
            model.init_kv_caches(device=device)
            with torch.no_grad():
                model.prefill_chunked(dummy_prompt)

            # Now measure forward_block with k new tokens
            new_tokens = torch.randint(
                10, model.config['vocab_size'], (1, k),
                device=device, dtype=torch.long
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                model.forward_block(new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()

            if trial >= warmup:
                times.append(t1 - t0)

        avg_time = sum(times) / len(times)
        v_table[k] = avg_time

    # Compute β = (V(8) - V(1)) / 7  (marginal cost of each extra token)
    v1 = v_table.get(1, 0)
    v8 = v_table.get(8, 0)
    beta = (v8 - v1) / 7.0 if v1 > 0 else 0

    # Normalise β as fraction of V(1)
    beta_normalised = beta / v1 if v1 > 0 else 0

    result = {
        "context_len": context_len,
        "V": {str(k): v_table[k] for k in sorted(v_table.keys())},
        "V1_ms": v1 * 1000,
        "V8_ms": v8 * 1000,
        "beta_abs_ms": beta * 1000,
        "beta_normalised": beta_normalised,
    }

    # Decision rule
    if beta_normalised <= 0.10:
        result["decision"] = "VIABLE: beta <= 0.10, aim for 2.0x"
    elif beta_normalised <= 0.25:
        result["decision"] = "VIABLE_CAUTIOUS: 0.10 < beta <= 0.25, aim >= 1.5x with gamma <= 3"
    else:
        result["decision"] = "FIX_VERIFY: beta > 0.25, fix verify path first"

    return result


def thermal_soak_test(model, device='cuda', duration_sec=600, report_interval=30):
    """Run continuous decode for *duration_sec*, logging temps.

    Parameters
    ----------
    model : OPTModel
    device : str
    duration_sec : int
    report_interval : int  (seconds between log prints)

    Returns
    -------
    dict with test results
    """
    dummy = torch.randint(10, 1000, (1, 32), device=device, dtype=torch.long)
    model.reset_caches()
    model.init_kv_caches(device=device)

    start_time = time.time()
    steps = 0
    max_temp = 0.0
    temps = []
    throttled = False

    with torch.no_grad():
        model.prefill_chunked(dummy)
        next_tok = torch.tensor([[42]], device=device, dtype=torch.long)

        while time.time() - start_time < duration_sec:
            logits, _ = model.forward_block(next_tok)
            next_tok_id = logits[0, -1, :].argmax(dim=-1).item()
            next_tok = torch.tensor([[next_tok_id]], device=device, dtype=torch.long)
            steps += 1

            # Reset cache if full
            if model.kv_caches[0].length >= model.max_positions - 1:
                model.reset_caches()
                model.init_kv_caches(device=device)
                model.prefill_chunked(dummy)

            elapsed = time.time() - start_time
            if steps % 100 == 0 and elapsed > 0:
                tok_s = steps / elapsed
                # Try to read GPU temp (Jetson)
                temp = _read_gpu_temp()
                if temp is not None:
                    temps.append(temp)
                    max_temp = max(max_temp, temp)

                if int(elapsed) % report_interval < 2:
                    print("  [%.0fs] %d steps, %.1f tok/s, temp=%.1f C" %
                          (elapsed, steps, tok_s, temp or 0))

    total_time = time.time() - start_time

    return {
        "duration_sec": total_time,
        "total_steps": steps,
        "tok_per_sec": steps / total_time if total_time > 0 else 0,
        "max_temp_c": max_temp,
        "temps": temps[-20:],  # last 20 readings
        "throttled": throttled,
        "passed": not throttled,
    }


def _read_gpu_temp():
    """Read GPU temperature on Jetson (thermal zone) or return None."""
    thermal_paths = [
        '/sys/devices/virtual/thermal/thermal_zone1/temp',  # GPU on Jetson
        '/sys/devices/virtual/thermal/thermal_zone2/temp',
    ]
    for path in thermal_paths:
        try:
            with open(path, 'r') as f:
                return int(f.read().strip()) / 1000.0
        except (IOError, OSError, ValueError):
            continue
    return None
