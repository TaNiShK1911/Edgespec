#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_runner.py  --  EdgeSpec-H2O Full Demo Runner (WP7-WP9)
============================================================
Hashcode 14.0, Track 01, Target: Jetson Nano 4GB (SM 5.3)

Orchestrates:
  * WP7: Side-by-side Mode A vs Mode D comparison
  * WP8: Replay from saved session JSON (offline / CI mode)
  * WP9: Live telemetry dashboard launch + demo log analysis

Usage
-----
  # Side-by-side comparison on a single prompt:
  python tools/demo_runner.py --mode compare --prompt "172.16.0.1 GET /api/status"

  # Full demo: log analysis + comparison + telemetry:
  python tools/demo_runner.py --mode all --log-file data/sample.clf

  # Replay a previously saved session:
  python tools/demo_runner.py --mode replay --replay-session results/session_20240101.json

  # Just launch telemetry dashboard:
  python tools/demo_runner.py --mode telemetry

Python 3.6.9 compatible: no f-strings, no dataclasses, no walrus operator.
"""
from __future__ import print_function
import argparse
import json
import os
import sys
import time
import threading

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    print('[demo_runner] WARNING: torch not available -- running in analysis-only mode')

try:
    from edgespec.config import OPT_CONFIGS, GATE3
    _CFG_OK = True
except ImportError:
    _CFG_OK = False
    OPT_CONFIGS = {}
    GATE3 = {}

RESULTS_DIR = os.path.join(_REPO_ROOT, 'results')
DATA_DIR = os.path.join(_REPO_ROOT, 'data')
SESSION_VERSION = '1.0'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts():
    return time.strftime('%Y%m%d_%H%M%S')


def _ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d)


def _format_tok_s(tok_s):
    if tok_s >= 100:
        return '%.0f tok/s' % tok_s
    return '%.1f tok/s' % tok_s


def _load_model(model_size, device, weights=None, dtype=None):
    """Load OPT model: quantized if weights found, else random-init FP16."""
    if dtype is None:
        dtype = torch.float16 if _TORCH_OK else None
    key = 'opt-' + model_size
    config = OPT_CONFIGS.get(key, {})
    if not config:
        raise ValueError('Unknown model size: %s' % model_size)

    candidates = []
    if weights:
        candidates.append(weights)
    candidates.append(os.path.join(DATA_DIR, 'weights',
                                   'opt-%s-4bit.edgespec' % model_size))
    candidates.append(os.path.join(DATA_DIR, 'weights',
                                   'opt-%s.bin' % model_size))

    found = None
    for c in candidates:
        if c and os.path.isfile(c):
            found = c
            break

    if found:
        from edgespec.model.loader import load_model as _lm
        model = _lm(found, device=device)
        print('[demo_runner] Loaded weights: %s' % found)
    else:
        from edgespec.model.opt import OPTModel
        model = OPTModel(config, dtype=dtype).to(device)
        print('[demo_runner] No weights -- random-init model (correctness only)')

    model.eval()
    return model, config


def _tokenise(text, config):
    """Tokeniser using real BPE if available, with deterministic fallback."""
    try:
        from edgespec.tokenizer import get_tokenizer
        tok = get_tokenizer()
        return tok.encode(text)
    except Exception:
        pass
    vocab_size = config.get('vocab_size', 50272)
    ids = []
    for ch in text.encode('utf-8'):
        ids.append(10 + (ch * 7 + len(ids) * 3) % (vocab_size - 10))
    while len(ids) < 8:
        ids.append(10 + len(ids))
    return ids[:512]


# ---------------------------------------------------------------------------
# WP7: Side-by-side Mode A vs Mode D
# ---------------------------------------------------------------------------

def run_compare(model_size='125m', device='cpu', weights=None,
                prompt_text=None, max_new_tokens=64,
                gamma=4, lenience=1.0,
                n_hh=128, n_local=128, n_pin=4, decay=0.98,
                output_dir=None, session=None):
    """Run Mode A (baseline) vs Mode D (H2O+spec) and print side-by-side."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    _ensure_dir(output_dir)

    if session is not None:
        return _replay_compare(session)

    if not _TORCH_OK:
        print('[demo_runner] torch unavailable -- skipping model comparison')
        return {}

    model, config = _load_model(model_size, device, weights)

    if prompt_text is None:
        prompt_text = (
            '172.16.0.1 - - [01/Jan/2024:00:00:01 +0000] '
            '"GET /api/health HTTP/1.1" 200 512 "-" "EdgeSpec-Monitor/1.0"'
        )

    print('\n=== WP7: Side-by-Side Comparison ===')
    print('Prompt (%d chars): %s' % (len(prompt_text), prompt_text[:80]))
    sys.stdout.flush()

    ids = _tokenise(prompt_text, config)
    prompt_ids = torch.tensor([ids], device=device, dtype=torch.long)

    # Mode A
    print('\n[Mode A] Standard decode (baseline)...')
    sys.stdout.flush()
    model.init_kv_caches(device=device)
    t0 = time.time()
    with torch.no_grad():
        a_ids = model.generate_greedy(prompt_ids, max_new_tokens=max_new_tokens)
    t_a = time.time() - t0
    if isinstance(a_ids, (list, tuple)):
        a_n = len(a_ids) - len(ids)
    else:
        a_n = a_ids.shape[1] - prompt_ids.shape[1]
    a_tok_s = a_n / t_a if t_a > 0 else 0

    # Mode D
    print('[Mode D] H2O + speculative decode...')
    sys.stdout.flush()
    from edgespec.spec.drafter import SuffixDrafter
    from edgespec.spec.sampler import SpeculativeSampler
    from edgespec.spec.verifier import SpeculativeVerifier

    model.init_h2o_caches(
        n_hh=n_hh, n_local=n_local, n_pin=n_pin,
        n_scratch=gamma + 2, decay=decay, device=device
    )
    drafter = SuffixDrafter(max_n=4, selector='importance')
    sampler = SpeculativeSampler(mode='greedy', lenience=lenience)
    verifier = SpeculativeVerifier(model=model, drafter=drafter,
                                   sampler=sampler, gamma_fixed=gamma)
    t0 = time.time()
    with torch.no_grad():
        d_ids, d_stats = verifier.generate(prompt_ids,
                                            max_new_tokens=max_new_tokens)
    t_d = time.time() - t0
    if isinstance(d_ids, (list, tuple)):
        d_n = len(d_ids) - len(ids)
    else:
        d_n = d_ids.shape[1] - prompt_ids.shape[1]
    d_tok_s = d_n / t_d if t_d > 0 else 0
    speedup = d_tok_s / a_tok_s if a_tok_s > 0 else 0.0

    thr = GATE3 if GATE3 else {'speedup_pass': 1.5,
                                'speedup_target': 2.0,
                                'speedup_stretch': 2.8}
    if speedup >= thr.get('speedup_stretch', 2.8):
        tier = 'stretch'
    elif speedup >= thr.get('speedup_target', 2.0):
        tier = 'target'
    elif speedup >= thr.get('speedup_pass', 1.5):
        tier = 'pass'
    else:
        tier = 'below'

    result = {
        'session_version': SESSION_VERSION,
        'timestamp': _ts(),
        'prompt_text': prompt_text,
        'prompt_len': len(ids),
        'max_new_tokens': max_new_tokens,
        'model': 'opt-%s' % model_size,
        'device': device,
        'gamma': gamma,
        'lenience': lenience,
        'mode_A': {'tok_s': a_tok_s, 'n_generated': a_n, 'latency_s': t_a},
        'mode_D': {
            'tok_s': d_tok_s,
            'n_generated': d_n,
            'latency_s': t_d,
            'alpha_hat': d_stats.get('alpha_hat', 0),
            'mean_accepted_per_step': (
                d_stats.get('total_tokens', d_n) /
                max(1, d_stats.get('total_steps', 1))
            ),
        },
        'speedup': speedup,
        'tier': tier,
        'pass': tier in ('pass', 'target', 'stretch'),
    }

    _print_table(result)

    sf = os.path.join(output_dir, 'session_%s.json' % _ts())
    with open(sf, 'w') as f:
        json.dump(result, f, indent=2)
    print('\nSession saved: %s' % sf)

    # Automatically notify live telemetry dashboard
    try:
        try:
            from urllib.request import Request, urlopen
        except ImportError:
            from urllib2 import Request, urlopen
        req = Request(
            'http://127.0.0.1:8765/api/replay',
            data=json.dumps({'session': result}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        urlopen(req, timeout=1)
    except Exception:
        pass

    return result


def _print_table(r):
    a = r['mode_A']
    d = r['mode_D']
    marks = {'pass': '[PASS]', 'target': '[TARGET]',
             'stretch': '[STRETCH]', 'below': '[BELOW]'}
    marker = marks.get(r['tier'], '[?]')
    print('')
    print('  %-22s  %12s  %12s' % ('Metric', 'Mode A', 'Mode D (H2O+Spec)'))
    print('  ' + '-' * 52)
    print('  %-22s  %12s  %12s' % ('Throughput',
          _format_tok_s(a['tok_s']), _format_tok_s(d['tok_s'])))
    print('  %-22s  %12.2f  %12.2f' % ('Latency (s)',
          a['latency_s'], d['latency_s']))
    print('  %-22s  %12d  %12d' % ('Tokens generated',
          a['n_generated'], d['n_generated']))
    print('  %-22s  %12s  %12.3f' % ('alpha_hat', '-', d.get('alpha_hat', 0)))
    print('  %-22s  %12s  %12.2f' % ('accepted/step', '-',
          d.get('mean_accepted_per_step', 0)))
    print('  ' + '-' * 52)
    print('  %-22s  %12s  %12.2f  %s' % ('Speedup', '', r['speedup'], marker))
    print('')


# ---------------------------------------------------------------------------
# WP8: Replay mode
# ---------------------------------------------------------------------------

def run_replay(session_file, output_dir=None):
    """Load a saved session JSON and re-display results."""
    if not os.path.isfile(session_file):
        print('[demo_runner] ERROR: session not found: %s' % session_file)
        return {}

    with open(session_file) as f:
        session = json.load(f)

    print('\n=== WP8: Replay Mode ===')
    print('File:     %s' % session_file)
    print('Version:  %s' % session.get('session_version', '?'))
    print('Recorded: %s' % session.get('timestamp', '?'))
    print('Model:    %s on %s' % (session.get('model', '?'),
                                   session.get('device', '?')))
    if 'prompt_text' in session:
        print('Prompt:   %s' % str(session['prompt_text'])[:80])

    if 'mode_A' in session and 'mode_D' in session:
        _print_table(session)
    elif 'results' in session:
        _print_bench(session)
    else:
        print(json.dumps(session, indent=2)[:2000])

    return session


def _replay_compare(session):
    print('\n=== WP8: Replay (in-memory) ===')
    if 'mode_A' in session and 'mode_D' in session:
        _print_table(session)
    return session


def _print_bench(session):
    results = session.get('results', [])
    print('\n  %-20s  %-6s  %10s  %8s' % ('Prompt', 'Mode', 'tok/s', 'alpha'))
    print('  ' + '-' * 50)
    for r in results:
        print('  %-20s  %-6s  %10.1f  %8.3f' % (
            str(r.get('prompt_name', '?'))[:20],
            r.get('mode', '?'),
            r.get('decode_tok_s', 0),
            r.get('alpha_hat', 0),
        ))


# ---------------------------------------------------------------------------
# WP9: Telemetry + log pipeline
# ---------------------------------------------------------------------------

def run_telemetry(port=8765, open_browser=True):
    """Start the SSE telemetry server in a daemon thread."""
    print('\n=== WP9: Telemetry Dashboard ===')
    try:
        from edgespec.telemetry.server import TelemetryServer
        server = TelemetryServer(port=port)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        url = 'http://127.0.0.1:%d' % port
        print('Dashboard: %s' % url)
        if open_browser:
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception:
                pass
        return server
    except Exception as e:
        print('[demo_runner] WARNING: telemetry server failed: %s' % e)
        return None


def run_log_analysis(log_file, output_dir=None, max_digests=5):
    """Run the CLF log pipeline: csv_to_clf -> build_digests."""
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, 'log_%s' % _ts())
    _ensure_dir(output_dir)

    print('\n=== WP9: Log Analysis Pipeline ===')
    print('Input:  %s' % log_file)
    print('Output: %s' % output_dir)

    steps = [
        {
            'name': 'csv_to_clf',
            'script': os.path.join(_REPO_ROOT, 'tools', 'csv_to_clf.py'),
            'args': [log_file, '--output-dir', output_dir],
        },
        {
            'name': 'build_digests',
            'script': os.path.join(_REPO_ROOT, 'tools', 'build_digests.py'),
            'args': ['--input-dir', output_dir, '--output-dir', output_dir,
                     '--max-digests', str(max_digests)],
        },
    ]

    all_ok = True
    for step in steps:
        script = step['script']
        if not os.path.isfile(script):
            print('  [SKIP] %s -- script not found' % step['name'])
            continue
        cmd = ' '.join([sys.executable, script] + step['args'])
        print('\n  $ %s' % cmd)
        ret = os.system(cmd)
        if ret != 0:
            print('  [FAIL] %s (exit %d)' % (step['name'], ret))
            all_ok = False
        else:
            print('  [OK] %s' % step['name'])

    return all_ok


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

def run_all(model_size='125m', device='cpu', weights=None,
            prompt_text=None, log_file=None,
            max_new_tokens=64, gamma=4, lenience=1.0,
            n_hh=128, n_local=128, n_pin=4, decay=0.98,
            telemetry_port=8765, open_browser=True,
            output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, 'demo_%s' % _ts())
    _ensure_dir(output_dir)

    print('\n' + '=' * 60)
    print('  EdgeSpec-H2O Full Demo  (Hashcode 14.0 / Track 01)')
    print('  Model: opt-%s  Device: %s' % (model_size, device))
    print('=' * 60)

    telemetry_server = run_telemetry(port=telemetry_port,
                                     open_browser=open_browser)

    if log_file:
        if os.path.isfile(log_file):
            run_log_analysis(log_file, output_dir=output_dir)
        else:
            print('[demo_runner] WARNING: log file not found: %s' % log_file)

    result = {}
    if _TORCH_OK and _CFG_OK:
        result = run_compare(
            model_size=model_size, device=device, weights=weights,
            prompt_text=prompt_text, max_new_tokens=max_new_tokens,
            gamma=gamma, lenience=lenience,
            n_hh=n_hh, n_local=n_local, n_pin=n_pin, decay=decay,
            output_dir=output_dir,
        )
        thr = GATE3 if GATE3 else {}
        print('\n=== Demo Summary ===')
        print('  Speedup: %.2fx   Tier: %s' % (
            result.get('speedup', 0), result.get('tier', '?').upper()))
        print('  Thresholds  pass=%.1fx  target=%.1fx  stretch=%.1fx' % (
            thr.get('speedup_pass', 1.5),
            thr.get('speedup_target', 2.0),
            thr.get('speedup_stretch', 2.8),
        ))
    else:
        print('[demo_runner] Model comparison skipped (torch/config unavailable)')

    if telemetry_server is not None:
        print('\nTelemetry running. Press Ctrl+C to exit.')
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print('\nShutting down.')
            telemetry_server.shutdown()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='EdgeSpec-H2O Demo Runner (WP7-WP9)')
    p.add_argument('--mode', choices=['compare', 'replay', 'telemetry', 'log', 'all'],
                   default='all')
    p.add_argument('--model-size', default='125m',
                   choices=['125m', '350m', '1.3b'])
    p.add_argument('--device', default=None)
    p.add_argument('--weights', default=None)
    p.add_argument('--prompt', default=None)
    p.add_argument('--log-file', default=None)
    p.add_argument('--replay-session', default=None)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--gamma', type=int, default=4)
    p.add_argument('--lenience', type=float, default=1.0)
    p.add_argument('--n-hh', type=int, default=128)
    p.add_argument('--n-local', type=int, default=128)
    p.add_argument('--n-pin', type=int, default=4)
    p.add_argument('--decay', type=float, default=0.98)
    p.add_argument('--telemetry-port', type=int, default=8765)
    p.add_argument('--no-browser', action='store_true')
    p.add_argument('--output-dir', default=None)
    args = p.parse_args()

    dev = args.device
    if dev is None:
        dev = 'cuda' if (_TORCH_OK and torch.cuda.is_available()) else 'cpu'

    if args.mode == 'compare':
        run_compare(model_size=args.model_size, device=dev,
                    weights=args.weights, prompt_text=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    gamma=args.gamma, lenience=args.lenience,
                    n_hh=args.n_hh, n_local=args.n_local,
                    n_pin=args.n_pin, decay=args.decay,
                    output_dir=args.output_dir)

    elif args.mode == 'replay':
        if not args.replay_session:
            print('ERROR: --replay-session required')
            sys.exit(1)
        run_replay(args.replay_session, output_dir=args.output_dir)

    elif args.mode == 'telemetry':
        srv = run_telemetry(port=args.telemetry_port,
                            open_browser=not args.no_browser)
        if srv:
            print('Press Ctrl+C to stop.')
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                srv.shutdown()

    elif args.mode == 'log':
        lf = args.log_file or os.path.join(DATA_DIR, 'sample.clf')
        run_log_analysis(lf, output_dir=args.output_dir)

    else:
        run_all(model_size=args.model_size, device=dev,
                weights=args.weights, prompt_text=args.prompt,
                log_file=args.log_file,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma, lenience=args.lenience,
                n_hh=args.n_hh, n_local=args.n_local,
                n_pin=args.n_pin, decay=args.decay,
                telemetry_port=args.telemetry_port,
                open_browser=not args.no_browser,
                output_dir=args.output_dir)


if __name__ == '__main__':
    main()
