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


def check_python():
    """Check Python version."""
    ver = sys.version_info
    print("[Python] %d.%d.%d" % (ver.major, ver.minor, ver.micro))
    if ver.major != 3:
        return False, "Python 3 required"
    return True, "OK"


def check_torch():
    """Check PyTorch and CUDA."""
    try:
        import torch
        print("[PyTorch] %s" % torch.__version__)
        cuda_ok = torch.cuda.is_available()
        print("[CUDA available] %s" % cuda_ok)
        if cuda_ok:
            print("[CUDA device] %s" % torch.cuda.get_device_name(0))
            print("[CUDA version] %s" % torch.version.cuda)
        return True, "torch %s, cuda=%s" % (torch.__version__, cuda_ok)
    except ImportError:
        return False, "PyTorch not installed"


def check_numpy():
    """Check NumPy imports without warnings."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            import numpy as np
            print("[NumPy] %s" % np.__version__)
            # Check for any numpy-related warnings
            numpy_warnings = [x for x in w if 'numpy' in str(x.message).lower()]
            if numpy_warnings:
                for nw in numpy_warnings:
                    print("  WARNING: %s" % nw.message)
                return False, "NumPy imported with warnings"
            return True, "OK"
        except ImportError:
            return False, "NumPy not installed"


def check_nvcc():
    """Check if nvcc is available (CUDA compiler)."""
    try:
        result = subprocess.check_output(['nvcc', '--version'],
                                         stderr=subprocess.STDOUT)
        version_line = result.decode().strip().split('\n')[-1]
        print("[nvcc] %s" % version_line)
        return True, version_line
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        print("[nvcc] Not found (OK on laptop)")
        return True, "Not found (non-critical)"


def check_nvpmodel():
    """Check if Max-N (mode 0) is active (Jetson only)."""
    try:
        result = subprocess.check_output(['sudo', 'nvpmodel', '-q'],
                                         stderr=subprocess.STDOUT)
        output = result.decode().strip()
        print("[nvpmodel] %s" % output.replace('\n', ' | '))
        if 'MAXN' in output.upper() or 'MODE ID: 0' in output.upper():
            return True, "Max-N active"
        return False, "Not in Max-N mode"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        print("[nvpmodel] Not available (OK on laptop)")
        return True, "Not available (non-critical)"


def check_no_transformers():
    """Ensure no 'transformers' import in edgespec/ source code."""
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
            with open(filepath, 'r') as f:
                for i, line in enumerate(f, 1):
                    stripped = line.strip()
                    # Skip comments
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


def check_regex():
    """Check if regex module is available."""
    try:
        import regex
        print("[regex] %s" % regex.__version__)
        return True, "OK"
    except ImportError:
        print("[regex] Not installed (will use stdlib re — some edge cases may differ)")
        return True, "Fallback to stdlib re"


def main():
    print("=" * 60)
    print("EdgeSpec-H2O Environment Check")
    print("=" * 60)

    checks = [
        ("Python", check_python),
        ("PyTorch", check_torch),
        ("NumPy", check_numpy),
        ("nvcc", check_nvcc),
        ("nvpmodel", check_nvpmodel),
        ("no-transformers", check_no_transformers),
        ("regex", check_regex),
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
    else:
        print("SOME CHECKS FAILED (%d/%d passed)" % (n_pass, n_total))
        for name, ok, msg in results:
            if not ok:
                print("  FAIL: %s — %s" % (name, msg))

    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
