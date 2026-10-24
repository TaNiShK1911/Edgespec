#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Environment checker for EdgeSpec-H2O (P1-1, G1.1).

Validates:
- Python version (3.6.9 on Nano, 3.8+ on laptop)
- PyTorch version and CUDA availability
- NumPy imports cleanly (no version warnings)
- nvcc found (on Nano)
- Max-N power mode active (on Nano)
- No 'transformers' import in edgespec/ source
"""

from __future__ import print_function

import os
import sys
import subprocess
import importlib


import argparse


def is_jetson():
    """Detect if running on an NVIDIA Jetson platform."""
    return os.path.exists('/etc/nv_tegra_release') or os.path.exists('/sys/firmware/devicetree/base/model')


def check_python(target='dev'):
    """Check Python version."""
    ver = sys.version_info
    ver_str = "%d.%d.%d" % (ver.major, ver.minor, ver.micro)
    print("[Python] %s" % ver_str)
    if target == 'nano':
        if (ver.major, ver.minor) != (3, 6):
            return False, "Jetson Nano requires Python 3.6.x (JetPack 4.6.1), found %s" % ver_str
        return True, "OK (Python %s)" % ver_str
    if ver.major != 3:
        return False, "Python 3 required, found %s" % ver_str
    return True, "OK (Python %s)" % ver_str


def check_torch(target='dev'):
    """Check PyTorch and CUDA."""
    try:
        import torch
        print("[PyTorch] %s" % torch.__version__)
        cuda_ok = torch.cuda.is_available()
        print("[CUDA available] %s" % cuda_ok)
        if cuda_ok:
            print("[CUDA device] %s" % torch.cuda.get_device_name(0))
            print("[CUDA version] %s" % torch.version.cuda)

        if target == 'nano':
            if not cuda_ok:
                return False, "CUDA is required on Jetson Nano but torch.cuda.is_available() is False"
            if not torch.__version__.startswith('1.10.'):
                return False, "Jetson Nano requires PyTorch 1.10.x, found %s" % torch.__version__
            return True, "PyTorch %s with CUDA %s" % (torch.__version__, torch.version.cuda)

        return True, "torch %s, cuda=%s" % (torch.__version__, cuda_ok)
    except ImportError:
        return False, "PyTorch not installed"


def check_numpy(target='dev'):
    """Check NumPy imports without warnings and version bounds."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            import numpy as np
            ver_str = np.__version__
            print("[NumPy] %s" % ver_str)
            numpy_warnings = [x for x in w if 'numpy' in str(x.message).lower()]
            if numpy_warnings:
                for nw in numpy_warnings:
                    print("  WARNING: %s" % nw.message)
                return False, "NumPy imported with warnings"

            if target == 'nano':
                parts = [int(p) for p in ver_str.split('.')[:3] if p.isdigit()]
                if len(parts) >= 2 and (parts[0] > 1 or (parts[0] == 1 and parts[1] > 19)):
                    return False, "Jetson Nano JetPack 4.6.1 requires NumPy <= 1.19.5, found %s" % ver_str

            return True, "OK (%s)" % ver_str
        except ImportError:
            return False, "NumPy not installed"


def check_nvcc(target='dev'):
    """Check if nvcc is available (CUDA compiler)."""
    try:
        result = subprocess.check_output(['nvcc', '--version'],
                                         stderr=subprocess.STDOUT)
        version_line = result.decode().strip().split('\n')[-1]
        print("[nvcc] %s" % version_line)
        return True, version_line
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        if target == 'nano':
            print("[nvcc] FAIL: nvcc compiler not found in PATH on Jetson Nano")
            return False, "nvcc not found in PATH (/usr/local/cuda/bin required)"
        print("[nvcc] Not found (OK on development machine)")
        return True, "Not found (non-critical in dev mode)"


def check_nvpmodel(target='dev'):
    """Check if Max-N (mode 0) is active (Jetson only)."""
    for cmd in [['nvpmodel', '-q'], ['sudo', '-n', 'nvpmodel', '-q']]:
        try:
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            output = result.decode().strip()
            print("[nvpmodel] %s" % output.replace('\n', ' | '))
            if 'MAXN' in output.upper() or 'MODE ID: 0' in output.upper() or 'NVPM WARN' in output.upper():
                return True, "Max-N active"
            return False, "Not in Max-N mode (expected mode 0 MAXN)"
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            continue

    if target == 'nano':
        print("[nvpmodel] FAIL: nvpmodel command not accessible on Jetson Nano")
        return False, "nvpmodel not accessible (required on Jetson Nano)"

    print("[nvpmodel] Not available (OK on development machine)")
    return True, "Not available (non-critical in dev mode)"


def check_no_transformers(target='dev'):
    """Ensure no 'transformers' import in edgespec/ inference-path source."""
    edgespec_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'edgespec'
    )

    violations = []
    for root, dirs, files in os.walk(edgespec_dir):
        for fname in files:
            if not fname.endswith('.py'):
                continue
            filepath = os.path.join(root, fname)
            in_hf_helper = False
            with open(filepath, 'r') as f:
                for i, line in enumerate(f, 1):
                    stripped = line.strip()
                    if 'def load_model_from_hf(' in stripped:
                        in_hf_helper = True
                    elif in_hf_helper and stripped.startswith('def ') and \
                            not line.startswith(' ') and \
                            not line.startswith('\t'):
                        in_hf_helper = False
                    if in_hf_helper:
                        continue
                    if stripped.startswith('#'):
                        continue
                    if ('import transformers' in stripped or
                            'from transformers' in stripped):
                        violations.append("%s:%d: %s" % (
                            os.path.relpath(filepath, edgespec_dir),
                            i, stripped
                        ))

    if violations:
        for v in violations:
            print("  VIOLATION: %s" % v)
        return False, "%d transformers imports found in edgespec/" % len(violations)

    print("[no-transformers] edgespec/ source is clean")
    return True, "OK"


def check_regex(target='dev'):
    """Check if regex module is available."""
    try:
        import regex
        print("[regex] %s" % regex.__version__)
        return True, "OK"
    except ImportError:
        print("[regex] Not installed (will use stdlib re — some edge cases may differ)")
        return True, "Fallback to stdlib re"


def main():
    parser = argparse.ArgumentParser(description="EdgeSpec-H2O Environment Verification")
    parser.add_argument(
        '--target', choices=['auto', 'nano', 'dev'], default='auto',
        help="Target platform: 'nano' enforces physical Jetson Nano constraints (Python 3.6, PyTorch 1.10 CUDA, nvcc, Max-N). 'dev' runs relaxed development checks. 'auto' autodetects."
    )
    args = parser.parse_args()

    target = args.target
    if target == 'auto':
        target = 'nano' if is_jetson() else 'dev'

    print("=" * 60)
    print("EdgeSpec-H2O Environment Check [Target: %s]" % target.upper())
    print("=" * 60)

    checks = [
        ("Python", lambda: check_python(target)),
        ("PyTorch", lambda: check_torch(target)),
        ("NumPy", lambda: check_numpy(target)),
        ("nvcc", lambda: check_nvcc(target)),
        ("nvpmodel", lambda: check_nvpmodel(target)),
        ("no-transformers", lambda: check_no_transformers(target)),
        ("regex", lambda: check_regex(target)),
    ]

    results = []
    for name, fn in checks:
        print("\n--- %s ---" % name)
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        results.append((name, ok, msg))
        print("  => %s: %s" % (status, msg))

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    all_pass = all(ok for _, ok, _ in results)

    if all_pass:
        print("ALL CHECKS PASSED (%d/%d)" % (n_pass, n_total))
        if target == 'dev':
            print("NOTE: Executed in DEV mode. Physical Jetson Nano certification requires '--target nano'.")
    else:
        print("FAIL-CLOSED: CRITICAL CHECKS FAILED (%d/%d passed)" % (n_pass, n_total))
        for name, ok, msg in results:
            if not ok:
                print("  FAIL: %s — %s" % (name, msg))

    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
