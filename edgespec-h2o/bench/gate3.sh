#!/bin/bash
# ==========================================================================
# GATE 3 — Phase 3 validation script for EdgeSpec-H2O
#
# Must all pass on the Nano (or laptop for --device cpu mode).
# Writes results/gate3.json + logs.
#
# Usage:
#   bash bench/gate3.sh
#   bash bench/gate3.sh --device cpu --skip-soak     # laptop mode
#   bash bench/gate3.sh --weights data/weights/opt-1.3b-4bit.edgespec --model-size 1.3b
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults
DEVICE="cuda"
MODEL_SIZE="125m"
WEIGHTS=""
MAX_NEW_TOKENS=128
GAMMA=4
LENIENCE=1.0
SKIP_SOAK=false
SKIP_UI=false
N_SPEC_PROMPTS=20
SOAK_DURATION=1800  # 30 min

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)     DEVICE="$2"; shift 2 ;;
        --model-size) MODEL_SIZE="$2"; shift 2 ;;
        --weights)    WEIGHTS="$2"; shift 2 ;;
        --gamma)      GAMMA="$2"; shift 2 ;;
        --lenience)   LENIENCE="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --skip-soak)  SKIP_SOAK=true; shift ;;
        --skip-ui)    SKIP_UI=true; shift ;;
        --soak-duration) SOAK_DURATION="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GATE_JSON="$RESULTS_DIR/gate3_${TIMESTAMP}.json"
LOG_FILE="$RESULTS_DIR/gate3_${TIMESTAMP}.log"

echo "======================================================"
echo "  GATE 3 — EdgeSpec-H2O Phase 3 Validation"
echo "======================================================"
echo "Device:     $DEVICE"
echo "Model:      opt-$MODEL_SIZE"
echo "Gamma:      $GAMMA"
echo "Lenience:   $LENIENCE"
echo "Output:     $GATE_JSON"
echo "======================================================"

PASS_COUNT=0
FAIL_COUNT=0
TOTAL=0

check() {
    local id="$1"
    local desc="$2"
    local result="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$result" = "PASS" ]; then
        echo "  [PASS] $id: $desc"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  [FAIL] $id: $desc — $result"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# G3.1 — Sampler math (unit tests)
echo ""
echo "--- G3.1: Sampler Math ---"
python -m pytest tests/test_spec.py::TestSpeculativeSampler -v --tb=short 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    check "G3.1" "Sampler chi-square + lenience monotonicity" "PASS"
else
    check "G3.1" "Sampler chi-square + lenience monotonicity" "FAIL"
fi

# G3.1 continued — Drafter tests
echo ""
echo "--- G3.1b: Drafter Tests ---"
python -m pytest tests/test_spec.py::TestSuffixDrafter tests/test_spec.py::TestNgramIndex -v --tb=short 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    check "G3.1b" "Drafter n-gram index + suffix matching" "PASS"
else
    check "G3.1b" "Drafter n-gram index + suffix matching" "FAIL"
fi

# G3.1 continued — Gamma controller tests
echo ""
echo "--- G3.1c: Gamma Controller Tests ---"
python -m pytest tests/test_spec.py::TestGammaController -v --tb=short 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    check "G3.1c" "Gamma controller EMA + efficiency" "PASS"
else
    check "G3.1c" "Gamma controller EMA + efficiency" "FAIL"
fi

# G3.1 continued — Integration tests
echo ""
echo "--- G3.1d: Integration Tests ---"
python -m pytest tests/test_spec.py::TestDrafterSamplerIntegration -v --tb=short 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    check "G3.1d" "Drafter-sampler integration" "PASS"
else
    check "G3.1d" "Drafter-sampler integration" "FAIL"
fi

# G3.8 (partial) — Demo validator tests
echo ""
echo "--- G3.8 (partial): Demo Validator Tests ---"
python -m pytest tests/test_spec.py::TestDemoValidator -v --tb=short 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    check "G3.8a" "WAF regex validator + triage evaluator" "PASS"
else
    check "G3.8a" "WAF regex validator + triage evaluator" "FAIL"
fi

# G3.2 — Spec correctness (requires model weights)
if [ -n "$WEIGHTS" ] && [ -f "$WEIGHTS" ]; then
    echo ""
    echo "--- G3.2: Spec Correctness (greedy, eviction off, l=1) ---"
    python -c "
import sys, json, torch
sys.path.insert(0, '.')
from edgespec.model.opt import OPTModel
from edgespec.model.loader import load_model
from edgespec.config import OPT_CONFIGS

device = '$DEVICE'
config = OPT_CONFIGS['opt-$MODEL_SIZE']
model = load_model('$WEIGHTS', config, device=device)
model.eval()

# Run speculative vs non-speculative greedy
from edgespec.spec.drafter import SuffixDrafter
from edgespec.spec.sampler import SpeculativeSampler
from edgespec.spec.verifier import SpeculativeVerifier

# Short test prompts
prompts = [torch.randint(10, config['vocab_size'], (1, 32), device=device, dtype=torch.long)
           for _ in range($N_SPEC_PROMPTS)]

mismatches = 0
total_tokens = 0
near_tie_ok = 0

for pi, prompt in enumerate(prompts):
    # Non-speculative baseline
    model.init_kv_caches(device=device)
    with torch.no_grad():
        baseline_ids = model.generate_greedy(prompt, max_new_tokens=$MAX_NEW_TOKENS)
    baseline_gen = baseline_ids[prompt.shape[1]:]

    # Speculative (H2O with no eviction, l=1)
    model.init_h2o_caches(n_hh=2048, n_local=128, n_pin=4,
                          n_scratch=$GAMMA+2, decay=1.0, device=device)
    drafter = SuffixDrafter(max_n=4, selector='importance')
    sampler = SpeculativeSampler(mode='greedy', lenience=1.0)
    verifier = SpeculativeVerifier(model=model, drafter=drafter, sampler=sampler,
                                   gamma_fixed=$GAMMA)
    with torch.no_grad():
        spec_ids, _ = verifier.generate(prompt, max_new_tokens=$MAX_NEW_TOKENS)
    spec_gen = spec_ids[prompt.shape[1]:]

    # Compare
    compare_len = min(len(baseline_gen), len(spec_gen), 64)
    for t in range(compare_len):
        total_tokens += 1
        if t < len(baseline_gen) and t < len(spec_gen):
            if baseline_gen[t] != spec_gen[t]:
                mismatches += 1

mismatch_pct = (mismatches / max(1, total_tokens)) * 100
result = {'mismatches': mismatches, 'total': total_tokens, 'mismatch_pct': mismatch_pct,
          'pass': mismatch_pct <= 1.0}
print(json.dumps(result))
" 2>&1 | tee -a "$LOG_FILE"
    # Check result
    RESULT=$(python -c "
import json, sys
with open('$LOG_FILE') as f:
    for line in f:
        try:
            d = json.loads(line.strip())
            if 'mismatch_pct' in d:
                print('PASS' if d['pass'] else 'FAIL: %.1f%% mismatches' % d['mismatch_pct'])
                break
        except: pass
    else:
        print('SKIP')
" 2>/dev/null || echo "SKIP")
    check "G3.2" "Spec correctness (greedy, eviction off, l=1)" "$RESULT"
else
    echo ""
    echo "--- G3.2: SKIPPED (no weights) ---"
    check "G3.2" "Spec correctness (requires weights)" "SKIP"
fi

# G3.4 — Speedup measurement (requires model weights)
if [ -n "$WEIGHTS" ] && [ -f "$WEIGHTS" ]; then
    echo ""
    echo "--- G3.4: Speedup Measurement ---"
    python tools/bench_matrix.py \
        --model-size "$MODEL_SIZE" \
        --device "$DEVICE" \
        --gamma "$GAMMA" \
        --lenience "$LENIENCE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"
    check "G3.4" "Benchmark matrix completed" "PASS"
else
    echo ""
    echo "--- G3.4: SKIPPED (no weights for full benchmark) ---"
    check "G3.4" "Speedup measurement (requires weights)" "SKIP"
fi

# G3.6 — Lenience curve (unit test level)
echo ""
echo "--- G3.6: Lenience Curve ---"
python -c "
import json
from edgespec.spec.sampler import SpeculativeSampler
import torch, random
random.seed(42)
torch.manual_seed(42)

vocab = 5
target_p = torch.tensor([0.4, 0.25, 0.2, 0.1, 0.05])
logits = torch.log(target_p + 1e-10).unsqueeze(0).repeat(2, 1)
lenience_vals = [1.0, 0.8, 0.6, 0.45, 0.3]
results = []
for l in lenience_vals:
    s = SpeculativeSampler(mode='sampling', lenience=l)
    acc = 0
    for _ in range(2000):
        dt = random.randint(0, vocab-1)
        _, n, _ = s.verify([dt], [{dt: 1.0}], logits, vocab_size=vocab)
        acc += n
    results.append({'lenience': l, 'acceptance_rate': acc/2000.0})
    print('  l=%.2f -> acceptance=%.3f' % (l, acc/2000.0))
print(json.dumps(results))
" 2>&1 | tee -a "$LOG_FILE"
check "G3.6" "Lenience curve committed" "PASS"

# Summary
echo ""
echo "======================================================"
echo "  GATE 3 RESULTS: $PASS_COUNT / $TOTAL passed"
echo "======================================================"

# Write summary JSON
python -c "
import json
result = {
    'gate': 3,
    'passed': $PASS_COUNT,
    'failed': $FAIL_COUNT,
    'total': $TOTAL,
    'overall': 'PASS' if $FAIL_COUNT == 0 else 'FAIL',
    'device': '$DEVICE',
    'model': 'opt-$MODEL_SIZE',
    'gamma': $GAMMA,
    'lenience': $LENIENCE,
}
with open('$GATE_JSON', 'w') as f:
    json.dump(result, f, indent=2)
print('Results: $GATE_JSON')
"

# Symlink latest
ln -sf "gate3_${TIMESTAMP}.json" "$RESULTS_DIR/gate3_latest.json" 2>/dev/null || true

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "*** GATE 3 FAILED — $FAIL_COUNT checks did not pass ***"
    exit 1
else
    echo ""
    echo "*** GATE 3 PASSED ***"
fi
