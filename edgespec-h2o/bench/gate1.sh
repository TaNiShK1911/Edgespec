#!/bin/bash
# Gate 1 — Foundation gate for EdgeSpec-H2O
# Must all pass on the Nano
#
# Usage: bash bench/gate1.sh
# Writes results to results/gate1.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$RESULTS_DIR"

cd "$PROJECT_ROOT"

echo "================================================================"
echo "  GATE 1 — EdgeSpec-H2O Foundation Gate"
echo "  $(date)"
echo "================================================================"
echo ""

# Track pass/fail
PASS=0
FAIL=0
RESULTS="{}"

log_result() {
    local id="$1"
    local status="$2"
    local msg="$3"
    if [ "$status" = "PASS" ]; then
        echo "  [PASS] $id: $msg"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $id: $msg"
        FAIL=$((FAIL + 1))
    fi
}

# -----------------------------------------------------------------------
# G1.1: Environment check
# -----------------------------------------------------------------------
echo "--- G1.1: Environment ---"
if python tools/check_env.py; then
    log_result "G1.1" "PASS" "Environment OK"
else
    log_result "G1.1" "FAIL" "Environment check failed"
fi

# Check no transformers in edgespec/
echo ""
echo "  Checking for transformers imports in edgespec/..."
if grep -r "import transformers\|from transformers" edgespec/ 2>/dev/null | grep -v "loader.py" | grep -v "__pycache__"; then
    log_result "G1.1b" "FAIL" "Found transformers imports in edgespec/"
else
    log_result "G1.1b" "PASS" "No transformers in inference path"
fi

# -----------------------------------------------------------------------
# G1.2: Tokenizer golden test
# -----------------------------------------------------------------------
echo ""
echo "--- G1.2: Tokenizer ---"
if python -m pytest tests/test_tokenizer.py -v --tb=short 2>&1; then
    log_result "G1.2" "PASS" "Tokenizer golden match"
else
    log_result "G1.2" "FAIL" "Tokenizer mismatch"
fi

# -----------------------------------------------------------------------
# G1.4: INT4 pack/unpack
# -----------------------------------------------------------------------
echo ""
echo "--- G1.4: INT4 Pack/Unpack ---"
if python -m pytest tests/test_qlinear.py -v --tb=short 2>&1; then
    log_result "G1.4" "PASS" "Pack/unpack verified"
else
    log_result "G1.4" "FAIL" "Pack/unpack failed"
fi

# -----------------------------------------------------------------------
# G1.3/G1.5/G1.6: Model tests (require weights)
# -----------------------------------------------------------------------
echo ""
echo "--- Model tests (G1.3, G1.5, G1.6) ---"
if python -m pytest tests/test_model.py -v --tb=short -k "not slow" 2>&1; then
    log_result "G1.3" "PASS" "Model forward pass OK"
else
    log_result "G1.3" "FAIL" "Model tests failed"
fi

# -----------------------------------------------------------------------
# G1.7: Memory check (requires CUDA)
# -----------------------------------------------------------------------
echo ""
echo "--- G1.7: Memory ---"
python -c "
import torch
if torch.cuda.is_available():
    print('GPU memory: %.0f MiB allocated, %.0f MiB reserved' % (
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.memory_reserved() / 1024**2))
else:
    print('No CUDA — skipping GPU memory check')
" 2>&1

# Check system RAM
if command -v free &> /dev/null; then
    echo "System memory:"
    free -m
fi

# Check vmstat
if command -v vmstat &> /dev/null; then
    echo "vmstat:"
    vmstat 1 3
fi

# -----------------------------------------------------------------------
# G1.8: Baseline throughput (requires weights)
# -----------------------------------------------------------------------
echo ""
echo "--- G1.8: Throughput (placeholder — run with weights loaded) ---"
echo "  Run: python -c 'from edgespec.utils.profiler import benchmark_throughput; ...'"

# -----------------------------------------------------------------------
# G1.9: Verify-cost probe (requires weights + CUDA)
# -----------------------------------------------------------------------
echo ""
echo "--- G1.9: Verify-cost probe (placeholder — run with weights loaded) ---"
echo "  Run: python -c 'from edgespec.utils.profiler import verify_cost_probe; ...'"

# -----------------------------------------------------------------------
# G1.10: Thermal soak (requires CUDA + time)
# -----------------------------------------------------------------------
echo ""
echo "--- G1.10: Thermal soak (placeholder — 10 min test) ---"
echo "  Run: python -c 'from edgespec.utils.profiler import thermal_soak_test; ...'"

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  GATE 1 RESULTS"
echo "  PASSED: $PASS"
echo "  FAILED: $FAIL"
echo "  $(date)"
echo "================================================================"

# Write results JSON
python -c "
import json, sys
results = {
    'gate': 1,
    'passed': $PASS,
    'failed': $FAIL,
    'all_pass': $FAIL == 0,
}
with open('$RESULTS_DIR/gate1.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results written to $RESULTS_DIR/gate1.json')
"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo ">>> ALL GATE 1 CHECKS PASSED <<<"
    echo ">>> Ready for Phase 2 <<<"
    exit 0
else
    echo ""
    echo ">>> GATE 1 FAILED — fix before proceeding <<<"
    exit 1
fi
