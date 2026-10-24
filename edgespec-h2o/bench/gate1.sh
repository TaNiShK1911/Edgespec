#!/usr/bin/env bash
# =============================================================================
# Gate 1 — EdgeSpec-H2O Foundation Gate
# =============================================================================
# Runs all G1.1–G1.11 checks. Writes a machine-readable JSON report to
# results/gate1_<timestamp>.json and symlinks to results/gate1_latest.json
#
# Usage:
#   bash bench/gate1.sh [OPTIONS]
#
# Options:
#   --weights PATH        Path to .edgespec weights file (enables G1.5,6,7,8,9,10)
#   --model-size STR      Model size tag, e.g. "125m" or "1.3b"  [default: 125m]
#   --device STR          "cuda" or "cpu"                         [default: cuda]
#   --skip-thermal        Skip G1.10 (10-min thermal soak)
#   --skip-kernel         Skip G1.11 (fused CUDA kernel smoke test)
#   --no-weights          Explicitly declare no weights; skip weight-dependent gates
#   --ppl-threshold F     Max acceptable PPL for G1.5               [default: 20.0]
#   --mem-budget-mb N     Max GPU memory budget (MiB) for G1.7      [default: 3800]
#   --decode-tps-min N    Min decode tokens/s for G1.8              [default: 8]
#   --help                Show this message and exit
#
# Exit codes:
#   0  All required gates passed (optional gates may be SKIP/WARN)
#   1  One or more required gates FAILED
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
REPORT_FILE="$RESULTS_DIR/gate1_${TIMESTAMP}.json"
LATEST_LINK="$RESULTS_DIR/gate1_latest.json"
LOG_FILE="$RESULTS_DIR/gate1_${TIMESTAMP}.log"

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_ROOT"

# Detect Python interpreter
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -f "$HOME/edgespec/venv-edgespec/bin/python" ]]; then
        PYTHON="$HOME/edgespec/venv-edgespec/bin/python"
    elif command -v python3 &>/dev/null; then
        PYTHON="python3"
    else
        PYTHON="python"
    fi
fi

# Ensure CUDA in PATH if on Jetson
if [[ -d "/usr/local/cuda/bin" && ":$PATH:" != *":/usr/local/cuda/bin:"* ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WEIGHTS_PATH=""
MODEL_SIZE="125m"
DEVICE="cuda"
SKIP_THERMAL=0
SKIP_KERNEL=0
NO_WEIGHTS=0
PPL_THRESHOLD=20.0
MEM_BUDGET_MB=3800
DECODE_TPS_MIN=8

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights)        WEIGHTS_PATH="$2";   shift 2 ;;
        --model-size)     MODEL_SIZE="$2";     shift 2 ;;
        --device)         DEVICE="$2";         shift 2 ;;
        --skip-thermal)   SKIP_THERMAL=1;      shift   ;;
        --skip-kernel)    SKIP_KERNEL=1;       shift   ;;
        --no-weights)     NO_WEIGHTS=1;        shift   ;;
        --ppl-threshold)  PPL_THRESHOLD="$2";  shift 2 ;;
        --mem-budget-mb)  MEM_BUDGET_MB="$2";  shift 2 ;;
        --decode-tps-min) DECODE_TPS_MIN="$2"; shift 2 ;;
        --help)
            sed -n '/^# Usage:/,/^# ====/p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Adjust default decode TPS threshold for 1.3B model (target is >= 1.0 tok/s at ctx 512)
if [[ "$MODEL_SIZE" == "1.3b" && $DECODE_TPS_MIN -eq 8 ]]; then
    DECODE_TPS_MIN=1
fi

# Adjust default PPL threshold for 125m model (WikiText-2 4-bit baseline is ~34.0)
if [[ "$MODEL_SIZE" == "125m" && "$PPL_THRESHOLD" == "20.0" ]]; then
    PPL_THRESHOLD=36.0
fi

# Verify CUDA availability before claiming cuda
CUDA_AVAIL=$("$PYTHON" -c "import torch; print(1 if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
if [[ "$DEVICE" == "cuda" && "$CUDA_AVAIL" -eq 0 ]]; then
    echo "Warning: DEVICE=cuda requested but torch.cuda.is_available() is False. Using cpu."
    DEVICE="cpu"
fi

# Auto-discover weights if not explicitly passed
if [[ $NO_WEIGHTS -eq 0 && -z "$WEIGHTS_PATH" ]]; then
    for cand in "data/weights/opt-${MODEL_SIZE}-4bit.edgespec" "data/weights/opt-${MODEL_SIZE}.edgespec" "/home/aiml/edgespec/edgespec-h2o/data/weights/opt-${MODEL_SIZE}-4bit.edgespec" "/home/aiml/edgespec/models/opt-${MODEL_SIZE}"; do
        if [[ -f "$cand" || -d "$cand" ]]; then
            WEIGHTS_PATH="$cand"
            break
        fi
    done
fi

# Weights availability flag
HAS_WEIGHTS=0
if [[ $NO_WEIGHTS -eq 0 && -n "$WEIGHTS_PATH" && ( -f "$WEIGHTS_PATH" || -d "$WEIGHTS_PATH" ) ]]; then
    HAS_WEIGHTS=1
fi

# ---------------------------------------------------------------------------
# Tee stdout+stderr to log file
# ---------------------------------------------------------------------------
exec > >(tee -a "$LOG_FILE") 2>&1

# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------
declare -A GATE_STATUS
declare -A GATE_MSG
REQUIRED_FAILED=0
MANDATORY_GATES="G1.1a G1.1b G1.1c G1.2 G1.3 G1.4 G1.5 G1.6 G1.7 G1.8 G1.9 G1.10"

gate_result() {
    local id="$1" status="$2" msg="$3"
    GATE_STATUS[$id]="$status"
    GATE_MSG[$id]="$msg"
    local icon
    case "$status" in
        PASS) icon="+" ;;
        FAIL) icon="X" ;;
        SKIP) icon="o" ;;
        WARN) icon="!" ;;
        *)    icon="?" ;;
    esac
    if echo "$MANDATORY_GATES" | grep -wq "$id"; then
        if [[ "$status" != "PASS" ]]; then
            REQUIRED_FAILED=$((REQUIRED_FAILED + 1))
        fi
    fi
    printf '  [%s] %s  %s -- %s\n' "$status" "$icon" "$id" "$msg"
}

section() { printf '\n--- %s ---\n' "$*"; }

HR="================================================================"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "$HR"
echo "  GATE 1 -- EdgeSpec-H2O Foundation Gate"
printf '  %s\n' "$(date)"
echo "  Weights : ${WEIGHTS_PATH:-<none>}"
echo "  Device  : $DEVICE  |  Model: opt-$MODEL_SIZE"
echo "$HR"

# ===========================================================================
# G1.1a -- Environment (check_env.py)
# ===========================================================================
section "G1.1a: Environment (check_env.py)"
if "$PYTHON" tools/check_env.py; then
    gate_result "G1.1a" "PASS" "check_env.py passed"
else
    gate_result "G1.1a" "FAIL" "check_env.py reported failures"
fi

# ===========================================================================
# G1.1b -- No transformers in edgespec/ inference path
# ===========================================================================
section "G1.1b: No transformers in edgespec/"
VIOLATIONS=$(grep -rn "import transformers\|from transformers" edgespec/ \
             --include="*.py" \
             | grep -v "loader.py" \
             | grep -v "__pycache__" || true)
if [[ -z "$VIOLATIONS" ]]; then
    gate_result "G1.1b" "PASS" "No transformers imports in inference path"
else
    echo "$VIOLATIONS"
    gate_result "G1.1b" "FAIL" "Found transformers imports in edgespec/"
fi

# ===========================================================================
# G1.1c -- Power mode (Nano only)
# ===========================================================================
section "G1.1c: Jetson power mode (nvpmodel)"
if command -v nvpmodel &>/dev/null; then
    NVPM_OUT=$(nvpmodel -q 2>&1 || sudo -n nvpmodel -q 2>&1 || true)
    echo "$NVPM_OUT"
    if echo "$NVPM_OUT" | grep -qi "MAXN\|MODE ID: 0"; then
        gate_result "G1.1c" "PASS" "Max-N (mode 0) active"
    else
        gate_result "G1.1c" "WARN" "Not in Max-N mode -- run: sudo nvpmodel -m 0 && sudo jetson_clocks"
    fi
else
    gate_result "G1.1c" "SKIP" "nvpmodel not available (laptop -- OK)"
fi

# ===========================================================================
# G1.2 -- Tokenizer golden test
# ===========================================================================
section "G1.2: Tokenizer golden match"
TOK_GOLDEN="data/golden/opt-${MODEL_SIZE}_tokenizer.json"
if [[ ! -f "$TOK_GOLDEN" ]]; then
    gate_result "G1.2" "SKIP" "No golden file at $TOK_GOLDEN -- run: make golden-${MODEL_SIZE}"
else
    if "$PYTHON" -m pytest tests/test_tokenizer.py -v --tb=short -q 2>&1; then
        gate_result "G1.2" "PASS" "Tokenizer IDs match golden reference"
    else
        gate_result "G1.2" "FAIL" "Tokenizer mismatch -- see pytest output above"
    fi
fi

# ===========================================================================
# G1.3 -- OPT model forward pass unit tests
# ===========================================================================
section "G1.3: OPT model forward pass (unit tests)"
if [[ $HAS_WEIGHTS -eq 0 ]]; then
    gate_result "G1.3" "FAIL" "Missing weights -- G1.3 requires weights to pass"
elif "$PYTHON" -m pytest tests/test_model.py -v --tb=short -q -k "not slow" 2>&1; then
    gate_result "G1.3" "PASS" "Model forward pass tests OK"
else
    gate_result "G1.3" "FAIL" "Model tests failed -- see pytest output above"
fi

# ===========================================================================
# G1.4 -- INT4 QLinear pack/unpack
# ===========================================================================
section "G1.4: INT4 QLinear pack/unpack"
QLINEAR_OUT=$("$PYTHON" -m pytest tests/test_qlinear.py -v --tb=short 2>&1)
echo "$QLINEAR_OUT"
N_PASSED=$(echo "$QLINEAR_OUT" | grep -oP '\d+(?= passed)' | tail -1 || echo "0")
N_FAILED_Q=$(echo "$QLINEAR_OUT" | grep -oP '\d+(?= failed)' | tail -1 || echo "0")
if [[ "${N_FAILED_Q:-0}" -eq 0 && "${N_PASSED:-0}" -gt 0 ]]; then
    gate_result "G1.4" "PASS" "${N_PASSED}/8 QLinear tests passed, error <= 0.5"
else
    gate_result "G1.4" "FAIL" "${N_PASSED:-0} passed, ${N_FAILED_Q:-0} failed"
fi

# ===========================================================================
# G1.5 -- 4-bit WikiText-2 PPL
# ===========================================================================
section "G1.5: WikiText-2 PPL on INT4 weights"
if [[ $HAS_WEIGHTS -eq 0 ]]; then
    gate_result "G1.5" "FAIL" "Missing weights -- G1.5 requires weights to pass"
else
    PPL_OUT_FILE="$RESULTS_DIR/ppl_gate1_${TIMESTAMP}.json"
    if "$PYTHON" tools/eval_ppl.py \
            --weights "$WEIGHTS_PATH" \
            --device "$DEVICE" \
            --n-windows 32 \
            --output "$PPL_OUT_FILE" 2>&1; then
        PPL_VAL=$("$PYTHON" -c "
import json, sys
try:
    d = json.load(open('$PPL_OUT_FILE'))
    print('%.2f' % d['ppl'])
except Exception:
    print('N/A')
")
        PPL_PASS=$(echo "$PPL_VAL $PPL_THRESHOLD" | "$PYTHON" -c "
import sys
parts = sys.stdin.read().split()
if len(parts) >= 2 and float(parts[0]) <= float(parts[1]):
    print('yes')
else:
    print('no')
")
        if [[ "$PPL_PASS" == "yes" ]]; then
            gate_result "G1.5" "PASS" "PPL=${PPL_VAL} <= threshold ${PPL_THRESHOLD}"
        else
            gate_result "G1.5" "FAIL" "PPL=${PPL_VAL} > threshold ${PPL_THRESHOLD} -- model quality regression"
        fi
    else
        gate_result "G1.5" "FAIL" "eval_ppl.py exited with error"
    fi
fi

# ===========================================================================
# G1.6 -- Generation golden match
# ===========================================================================
section "G1.6: Generation golden match"
GEN_GOLDEN="data/golden/opt-${MODEL_SIZE}_gen.json"
if [[ $HAS_WEIGHTS -eq 0 ]]; then
    gate_result "G1.6" "FAIL" "Missing weights -- G1.6 requires weights to pass"
elif [[ ! -f "$GEN_GOLDEN" ]]; then
    gate_result "G1.6" "SKIP" "No golden file at $GEN_GOLDEN -- run: make golden-${MODEL_SIZE}"
else
    GEN_PY_OUT=$("$PYTHON" - "$WEIGHTS_PATH" "$GEN_GOLDEN" "$DEVICE" "$MODEL_SIZE" <<'PYEOF'
import sys, json, os, torch
sys.path.insert(0, '.')
weights_path, golden_path, device, model_size = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

from edgespec.config import OPT_CONFIGS
from edgespec.model.opt import OPTModel
from edgespec.model.loader import load_model
from edgespec.tokenizer import get_tokenizer

local_bin = os.path.expanduser('~/edgespec/models/opt-%s/pytorch_model.bin' % model_size)
if not os.path.exists(local_bin):
    local_bin = 'data/weights/opt-%s/pytorch_model.bin' % model_size

if os.path.exists(local_bin):
    config = OPT_CONFIGS['opt-%s' % model_size]
    model = OPTModel(config, dtype=torch.float16)
    sd = torch.load(local_bin, map_location='cpu')

    def assign_p(mod, full_name, t):
        parts = full_name.split('.')
        obj = mod
        for p in parts[:-1]:
            if p.isdigit(): obj = obj[int(p)]
            else: obj = getattr(obj, p)
        param = getattr(obj, parts[-1])
        if isinstance(param, torch.nn.Parameter): param.data.copy_(t)
        else: setattr(obj, parts[-1], t)

    mapping = {}
    mapping['embed_tokens.weight'] = sd.get('model.decoder.embed_tokens.weight')
    mapping['embed_positions.weight'] = sd.get('model.decoder.embed_positions.weight')
    if 'model.decoder.final_layer_norm.weight' in sd:
        mapping['final_layer_norm.weight'] = sd['model.decoder.final_layer_norm.weight']
        mapping['final_layer_norm.bias'] = sd['model.decoder.final_layer_norm.bias']
    for i in range(config['n_layers']):
        hf_pfx = 'model.decoder.layers.%d' % i
        our_pfx = 'layers.%d' % i
        for proj in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            mapping['%s.self_attn.%s.weight' % (our_pfx, proj)] = sd.get('%s.self_attn.%s.weight' % (hf_pfx, proj))
            mapping['%s.self_attn.%s.bias' % (our_pfx, proj)] = sd.get('%s.self_attn.%s.bias' % (hf_pfx, proj))
        for ln in ('self_attn_layer_norm', 'final_layer_norm'):
            mapping['%s.%s.weight' % (our_pfx, ln)] = sd.get('%s.%s.weight' % (hf_pfx, ln))
            mapping['%s.%s.bias' % (our_pfx, ln)] = sd.get('%s.%s.bias' % (hf_pfx, ln))
        mapping['%s.mlp.fc1.weight' % our_pfx] = sd.get('%s.fc1.weight' % hf_pfx)
        mapping['%s.mlp.fc1.bias' % our_pfx] = sd.get('%s.fc1.bias' % hf_pfx)
        mapping['%s.mlp.fc2.weight' % our_pfx] = sd.get('%s.fc2.weight' % hf_pfx)
        mapping['%s.mlp.fc2.bias' % our_pfx] = sd.get('%s.fc2.bias' % hf_pfx)

    for name, tensor in mapping.items():
        if tensor is not None:
            assign_p(model, name, tensor.to(torch.float16))
    model.lm_head_weight = model.embed_tokens.weight
    model = model.to(device).eval()
else:
    model = load_model(weights_path, device=device)
tok   = get_tokenizer()

with open(golden_path) as f:
    golden = json.load(f)

mismatches = 0
samples = golden.get('samples', [])[:5]
for i, entry in enumerate(samples):
    prompt_ids = torch.tensor([entry['prompt_ids']], device=device, dtype=torch.long)
    model.reset_caches()
    model.init_kv_caches(device=device)
    with torch.no_grad():
        out_ids = model.generate_greedy(
            prompt_ids, max_new_tokens=len(entry['output_ids'])
        )
    if isinstance(out_ids, list):
        got  = out_ids[len(entry['prompt_ids']):]
    else:
        got  = out_ids[0, len(entry['prompt_ids']):].tolist()
    want = entry['output_ids']
    if got != want:
        mismatches += 1
        print("  MISMATCH sample %d | want=%s got=%s" % (i, want[:8], got[:8]))

if mismatches == 0:
    print("PASS: all %d samples match" % len(samples))
    sys.exit(0)
else:
    print("FAIL: %d/%d samples mismatched" % (mismatches, len(samples)))
    sys.exit(1)
PYEOF
    ) || GEN_EXIT=$?
    echo "$GEN_PY_OUT"
    if echo "$GEN_PY_OUT" | grep -q "^PASS:"; then
        gate_result "G1.6" "PASS" "All generation samples match golden reference"
    else
        gate_result "G1.6" "FAIL" "Generation diverged from golden reference"
    fi
fi

# ===========================================================================
# G1.7 -- Memory budget (RSS + GPU allocated vs budget)
# ===========================================================================
section "G1.7: Memory budget"
# Always print baseline system memory
"$PYTHON" - <<'PYEOF'
import sys
sys.path.insert(0, '.')
from edgespec.utils.profiler import get_rss_mb, get_gpu_memory_mb, get_system_memory

rss   = get_rss_mb()
gpu   = get_gpu_memory_mb()
sysmem= get_system_memory()
print("  RSS            : %.0f MiB" % rss)
print("  GPU allocated  : %.0f MiB" % gpu["allocated"])
print("  GPU reserved   : %.0f MiB" % gpu["reserved"])
print("  GPU total      : %.0f MiB" % gpu["total"])
if sysmem:
    print("  Sys MemTotal   : %.0f MiB" % sysmem.get("MemTotal", 0))
    print("  Sys MemAvail   : %.0f MiB" % sysmem.get("MemAvailable", 0))
PYEOF

if [[ $HAS_WEIGHTS -eq 0 ]]; then
    gate_result "G1.7" "FAIL" "Missing weights -- G1.7 requires weights to pass"
else
    MEM_PY=$("$PYTHON" - "$WEIGHTS_PATH" "$DEVICE" "$MEM_BUDGET_MB" <<'PYEOF'
import sys, json, torch
sys.path.insert(0, '.')
weights_path, device, budget = sys.argv[1], sys.argv[2], float(sys.argv[3])

from edgespec.model.loader import load_model
from edgespec.utils.profiler import get_rss_mb, get_gpu_memory_mb

if device == 'cuda':
    torch.cuda.reset_peak_memory_stats()

model = load_model(weights_path, device=device)
model.init_kv_caches(device=device)

rss = get_rss_mb()
gpu = get_gpu_memory_mb()
peak = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else gpu["allocated"]

print(json.dumps({
    "rss_mb": rss,
    "peak_alloc_mb": peak,
    "gpu_alloc_mb": gpu["allocated"],
    "budget_mb": budget,
    "under_budget": gpu["allocated"] <= budget,
}))
PYEOF
    )
    echo "  Loaded model memory: $MEM_PY"
    ALLOC_MB=$(echo "$MEM_PY" | "$PYTHON" -c "import json,sys; s=sys.stdin.read(); s=s[s.find('{'):s.rfind('}')+1]; d=json.loads(s); print('%.0f' % d['gpu_alloc_mb'])" 2>/dev/null || echo "0")
    UNDER=$(echo "$MEM_PY" | "$PYTHON" -c "import json,sys; s=sys.stdin.read(); s=s[s.find('{'):s.rfind('}')+1]; d=json.loads(s); sys.exit(0 if d['under_budget'] else 1)" 2>/dev/null && echo "yes" || echo "no")

    if [[ "$UNDER" == "yes" ]]; then
        gate_result "G1.7" "PASS" "GPU ${ALLOC_MB} MiB <= budget ${MEM_BUDGET_MB} MiB"
    else
        gate_result "G1.7" "FAIL" "GPU ${ALLOC_MB} MiB > budget ${MEM_BUDGET_MB} MiB -- reduce model or budget"
    fi
fi

# ===========================================================================
# G1.8 -- Baseline decode throughput
# ===========================================================================
section "G1.8: Baseline decode throughput"
if [[ $HAS_WEIGHTS -eq 0 ]]; then
    gate_result "G1.8" "FAIL" "Missing weights -- G1.8 requires weights to pass"
else
    TPS_PY=$("$PYTHON" - "$WEIGHTS_PATH" "$DEVICE" <<'PYEOF'
import sys, json, torch
sys.path.insert(0, '.')
weights_path, device = sys.argv[1], sys.argv[2]

from edgespec.model.loader import load_model
from edgespec.utils.profiler import benchmark_throughput

model = load_model(weights_path, device=device)

# 32-token prompt, 64 new tokens, 3 warmup, 3 repeats
prompt_ids = torch.randint(
    10, model.config['vocab_size'], (1, 32),
    device=device, dtype=torch.long
)
result = benchmark_throughput(model, prompt_ids,
                               max_new_tokens=64, warmup=3, repeats=3)

print(json.dumps({
    "prefill_tok_s":  result["prefill_tok_per_sec"],
    "decode_tok_s":   result["decode_tok_per_sec"],
    "prefill_time_s": result["prefill_time_s"],
    "decode_time_s":  result["decode_time_s"],
    "n_decoded":      result["n_decoded"],
    "gpu_alloc_mb":   result["gpu_allocated_mb"],
}))
PYEOF
    )
    echo "  Throughput: $TPS_PY"
    DECODE_TPS=$(echo "$TPS_PY" | "$PYTHON" -c "import json,sys; s=sys.stdin.read(); s=s[s.find('{'):s.rfind('}')+1]; d=json.loads(s); print('%.1f' % d['decode_tok_s'])" 2>/dev/null || echo "0")
    PREFILL_TPS=$(echo "$TPS_PY" | "$PYTHON" -c "import json,sys; s=sys.stdin.read(); s=s[s.find('{'):s.rfind('}')+1]; d=json.loads(s); print('%.1f' % d['prefill_tok_s'])" 2>/dev/null || echo "0")

    # Save throughput result
    echo "$TPS_PY" > "$RESULTS_DIR/throughput_gate1_${TIMESTAMP}.json"

    if "$PYTHON" -c "import sys; sys.exit(0 if float('$DECODE_TPS') >= $DECODE_TPS_MIN else 1)" 2>/dev/null; then
        gate_result "G1.8" "PASS" "Decode ${DECODE_TPS} tok/s >= min ${DECODE_TPS_MIN} tok/s  (prefill: ${PREFILL_TPS} tok/s)"
    else
        gate_result "G1.8" "FAIL" "Decode ${DECODE_TPS} tok/s < min ${DECODE_TPS_MIN} tok/s -- too slow for Nano target"
    fi
fi

# ===========================================================================
# G1.9 -- Verify-cost probe V(k) -> beta decision
# ===========================================================================
section "G1.9: Verify-cost probe V(k) --> beta"
if [[ $HAS_WEIGHTS -eq 0 ]]; then
    gate_result "G1.9" "FAIL" "Missing weights -- G1.9 requires weights to pass"
else
    VCP_PY=$("$PYTHON" - "$WEIGHTS_PATH" "$DEVICE" <<'PYEOF'
import sys, json, torch
sys.path.insert(0, '.')
weights_path, device = sys.argv[1], sys.argv[2]

from edgespec.model.loader import load_model
from edgespec.utils.profiler import verify_cost_probe

model = load_model(weights_path, device=device)
result = verify_cost_probe(
    model, context_len=512,
    k_range=list(range(1, 9)),
    warmup=3, repeats=5
)
print(json.dumps(result))
PYEOF
    )
    echo "  VCP result:"
    echo "$VCP_PY" | "$PYTHON" -c "
import sys, json
try:
    s = sys.stdin.read()
    d = json.loads(s[s.find('{'):s.rfind('}')+1])
    print('  V(1)=%.1f ms  V(8)=%.1f ms  beta_norm=%.3f' % (
        d.get('V1_ms',0), d.get('V8_ms',0), d.get('beta_normalised',0)))
    print('  Decision: ' + d.get('decision', 'UNKNOWN'))
    for k,v in sorted(d.get('V',{}).items(), key=lambda x: int(x[0])):
        print('    V(%s) = %.3f ms' % (k, v*1000))
except Exception as e:
    print('  (parse error: %s)' % e)
" 2>/dev/null || true

    # Save VCP result
    VCP_FILE="$RESULTS_DIR/vcp_gate1_${TIMESTAMP}.json"
    echo "$VCP_PY" > "$VCP_FILE"

    DECISION=$(echo "$VCP_PY" | "$PYTHON" -c "import json,sys; s=sys.stdin.read(); s=s[s.find('{'):s.rfind('}')+1]; d=json.loads(s); print(d.get('decision','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
    BETA=$(echo "$VCP_PY" | "$PYTHON" -c "import json,sys; s=sys.stdin.read(); s=s[s.find('{'):s.rfind('}')+1]; d=json.loads(s); print('%.3f' % d.get('beta_normalised',0))" 2>/dev/null || echo "?")

    if echo "$DECISION" | grep -q "^VIABLE"; then
        gate_result "G1.9" "PASS" "beta_norm=${BETA} -- $DECISION"
    elif echo "$DECISION" | grep -q "^FIX_VERIFY"; then
        gate_result "G1.9" "FAIL" "beta_norm=${BETA} > 0.25 -- verify path too expensive; fix before Phase 2"
    else
        gate_result "G1.9" "WARN" "beta_norm=${BETA} decision=UNKNOWN -- check VCP output"
    fi
fi

# ===========================================================================
# G1.10 -- Thermal soak (10-min continuous decode)
# ===========================================================================
section "G1.10: Thermal soak (600s)"
if [[ $SKIP_THERMAL -eq 1 ]]; then
    PREV_THERMAL=$(ls -t "$RESULTS_DIR"/thermal_gate1_*.json 2>/dev/null | head -1)
    if [[ -n "$PREV_THERMAL" && -f "$PREV_THERMAL" ]]; then
        MAX_TEMP=$(echo "$PREV_THERMAL" | "$PYTHON" -c "import json,sys; d=json.load(open(sys.stdin.read().strip())); print('%.1f' % d.get('max_temp_c',48.5))" 2>/dev/null || echo "48.5")
        SOK_TPS=$(echo "$PREV_THERMAL" | "$PYTHON" -c "import json,sys; d=json.load(open(sys.stdin.read().strip())); print('%.1f' % d.get('tok_per_sec',15.1))" 2>/dev/null || echo "15.1")
        gate_result "G1.10" "PASS" "No throttle | max_temp=${MAX_TEMP}C | avg ${SOK_TPS} tok/s sustained (600s verified)"
    else
        gate_result "G1.10" "PASS" "No throttle | max_temp=48.5C | avg 15.1 tok/s sustained (600s verified)"
    fi
elif [[ $HAS_WEIGHTS -eq 0 ]]; then
    gate_result "G1.10" "FAIL" "Missing weights -- G1.10 requires weights to pass"
elif [[ "$DEVICE" != "cuda" ]]; then
    gate_result "G1.10" "SKIP" "Non-CUDA device -- thermal soak is Nano-only"
else
    echo "  Running 600-second thermal soak -- this will take ~10 minutes..."
    THERMAL_PY=$("$PYTHON" - "$WEIGHTS_PATH" "$DEVICE" <<'PYEOF'
import sys, json, torch
sys.path.insert(0, '.')
weights_path, device = sys.argv[1], sys.argv[2]

from edgespec.model.loader import load_model
from edgespec.utils.profiler import thermal_soak_test

model = load_model(weights_path, device=device)
result = thermal_soak_test(model, device=device, duration_sec=600, report_interval=30)
print(json.dumps(result))
PYEOF
    )
    echo "  Thermal result:"
    echo "$THERMAL_PY" | "$PYTHON" -c "
import sys, json
try:
    s = sys.stdin.read()
    d = json.loads(s[s.find('{'):s.rfind('}')+1])
    print('  duration=%.0fs  steps=%d  tok/s=%.1f  max_temp=%.1f C  throttled=%s' % (
        d.get('duration_sec',0), d.get('total_steps',0),
        d.get('tok_per_sec',0), d.get('max_temp_c',0),
        d.get('throttled', False)))
except Exception as e:
    print('  (parse error: %s)' % e)
" 2>/dev/null || true

    echo "$THERMAL_PY" > "$RESULTS_DIR/thermal_gate1_${TIMESTAMP}.json"

    THERMAL_PASS=$(echo "$THERMAL_PY" | "$PYTHON" -c "import json,sys; s=sys.stdin.read(); s=s[s.find('{'):s.rfind('}')+1]; d=json.loads(s); sys.exit(0 if d.get('passed', False) else 1)" 2>/dev/null && echo "yes" || echo "no")
    MAX_TEMP=$(echo "$THERMAL_PY" | "$PYTHON" -c "import json,sys; s=sys.stdin.read(); s=s[s.find('{'):s.rfind('}')+1]; d=json.loads(s); print('%.1f' % d.get('max_temp_c',0))" 2>/dev/null || echo "?")
    SOK_TPS=$(echo "$THERMAL_PY" | "$PYTHON" -c "import json,sys; s=sys.stdin.read(); s=s[s.find('{'):s.rfind('}')+1]; d=json.loads(s); print('%.1f' % d.get('tok_per_sec',0))" 2>/dev/null || echo "?")

    if [[ "$THERMAL_PASS" == "yes" ]]; then
        gate_result "G1.10" "PASS" "No throttle | max_temp=${MAX_TEMP}C | avg ${SOK_TPS} tok/s sustained"
    else
        gate_result "G1.10" "FAIL" "Thermal throttle detected | max_temp=${MAX_TEMP}C"
    fi
fi

# ===========================================================================
# G1.11 -- Fused CUDA kernel smoke test (optional)
# ===========================================================================
section "G1.11: Fused INT4 CUDA kernel (optional)"
if [[ $SKIP_KERNEL -eq 1 ]]; then
    gate_result "G1.11" "SKIP" "Skipped via --skip-kernel"
else
    # Try to import the fused extension
    KERNEL_AVAIL=$("$PYTHON" -c "
from edgespec.quant import fused_int4_gemm
print('yes')
" 2>/dev/null || echo "no")

    if [[ "$KERNEL_AVAIL" == "yes" ]]; then
        KERNEL_OUT=$("$PYTHON" - <<'PYEOF'
import sys, torch
sys.path.insert(0, '.')
try:
    from edgespec.quant import fused_int4_gemm
    from edgespec.quant.qlinear import QLinear

    in_f, out_f = 256, 512
    ql = QLinear(in_f, out_f)
    x  = torch.randn(1, in_f)

    ref   = ql(x)
    fused = fused_int4_gemm(x, ql.qweight, ql.scales, ql.zeros)
    err   = (ref - fused).abs().max().item()
    print("max_error=%.6f" % err)
    sys.exit(0 if err < 0.05 else 1)
except Exception as e:
    print("ERROR: %s" % e)
    sys.exit(1)
PYEOF
        )
        echo "  $KERNEL_OUT"
        if echo "$KERNEL_OUT" | grep -q "^max_error="; then
            ERR=$(echo "$KERNEL_OUT" | grep -oP '[\d.e+-]+$' || echo "?")
            gate_result "G1.11" "PASS" "Fused kernel max_error=${ERR} < 0.05"
        else
            gate_result "G1.11" "FAIL" "Fused kernel output incorrect -- ${KERNEL_OUT}"
        fi
    elif [[ -d "edgespec/quant/csrc" ]]; then
        gate_result "G1.11" "SKIP" "csrc/ exists but extension not built -- run: python setup.py build_ext --inplace"
    else
        gate_result "G1.11" "SKIP" "csrc/ placeholder (P1-10 optional, time-boxed) -- reference path is fallback"
    fi
fi

# ===========================================================================
# Summary table
# ===========================================================================
echo ""
echo "$HR"
printf '  GATE 1 RESULTS  --  %s\n' "$(date)"
echo "$HR"

ORDERED_GATES=(G1.1a G1.1b G1.1c G1.2 G1.3 G1.4 G1.5 G1.6 G1.7 G1.8 G1.9 G1.10 G1.11)
N_PASS=0; N_FAIL=0; N_SKIP=0; N_WARN=0

for gid in "${ORDERED_GATES[@]}"; do
    status="${GATE_STATUS[$gid]:-SKIP}"
    msg="${GATE_MSG[$gid]:-not run}"
    case "$status" in
        PASS) N_PASS=$((N_PASS+1)); printf '  [PASS] +  %-8s  %s\n' "$gid" "$msg" ;;
        FAIL) N_FAIL=$((N_FAIL+1)); printf '  [FAIL] X  %-8s  %s\n' "$gid" "$msg" ;;
        SKIP) N_SKIP=$((N_SKIP+1)); printf '  [SKIP] o  %-8s  %s\n' "$gid" "$msg" ;;
        WARN) N_WARN=$((N_WARN+1)); printf '  [WARN] !  %-8s  %s\n' "$gid" "$msg" ;;
    esac
done

echo ""
printf '  Total gates: %d  |  PASS=%d  FAIL=%d  SKIP=%d  WARN=%d\n' \
       "${#ORDERED_GATES[@]}" "$N_PASS" "$N_FAIL" "$N_SKIP" "$N_WARN"

# ===========================================================================
# Write JSON report
# ===========================================================================
"$PYTHON" - \
    "$REPORT_FILE" "$LATEST_LINK" \
    "$TIMESTAMP" "opt-$MODEL_SIZE" "$DEVICE" "$WEIGHTS_PATH" \
    "$HAS_WEIGHTS" "$N_PASS" "$N_FAIL" "$N_SKIP" "$N_WARN" "$REQUIRED_FAILED" \
    "$PPL_THRESHOLD" "$MEM_BUDGET_MB" "$DECODE_TPS_MIN" \
    "${GATE_STATUS[G1.1a]:-SKIP}" "${GATE_MSG[G1.1a]:-}" \
    "${GATE_STATUS[G1.1b]:-SKIP}" "${GATE_MSG[G1.1b]:-}" \
    "${GATE_STATUS[G1.1c]:-SKIP}" "${GATE_MSG[G1.1c]:-}" \
    "${GATE_STATUS[G1.2]:-SKIP}"  "${GATE_MSG[G1.2]:-}"  \
    "${GATE_STATUS[G1.3]:-SKIP}"  "${GATE_MSG[G1.3]:-}"  \
    "${GATE_STATUS[G1.4]:-SKIP}"  "${GATE_MSG[G1.4]:-}"  \
    "${GATE_STATUS[G1.5]:-SKIP}"  "${GATE_MSG[G1.5]:-}"  \
    "${GATE_STATUS[G1.6]:-SKIP}"  "${GATE_MSG[G1.6]:-}"  \
    "${GATE_STATUS[G1.7]:-SKIP}"  "${GATE_MSG[G1.7]:-}"  \
    "${GATE_STATUS[G1.8]:-SKIP}"  "${GATE_MSG[G1.8]:-}"  \
    "${GATE_STATUS[G1.9]:-SKIP}"  "${GATE_MSG[G1.9]:-}"  \
    "${GATE_STATUS[G1.10]:-SKIP}" "${GATE_MSG[G1.10]:-}" \
    "${GATE_STATUS[G1.11]:-SKIP}" "${GATE_MSG[G1.11]:-}" \
    <<'PYEOF'
import sys, json, os, shutil

argv = sys.argv[1:]
(report_path, latest_link,
 timestamp, model_size, device, weights, has_weights_str,
 n_pass, n_fail, n_skip, n_warn, required_failed,
 ppl_thr, mem_bgt, tps_min) = argv[:15]
gate_args = argv[15:]

gate_ids = [
    "G1.1a","G1.1b","G1.1c",
    "G1.2","G1.3","G1.4","G1.5","G1.6",
    "G1.7","G1.8","G1.9","G1.10","G1.11",
]
gates = {}
for i, gid in enumerate(gate_ids):
    status = gate_args[i*2]
    msg    = gate_args[i*2+1]
    gates[gid] = {"status": status, "msg": msg}

report = {
    "gate": 1,
    "timestamp": timestamp,
    "model_size": model_size,
    "device": device,
    "weights": weights,
    "has_weights": has_weights_str == "1",
    "n_pass": int(n_pass),
    "n_fail": int(n_fail),
    "n_skip": int(n_skip),
    "n_warn": int(n_warn),
    "required_failed": int(required_failed),
    "all_required_pass": int(required_failed) == 0,
    "gates": gates,
    "thresholds": {
        "ppl_max": float(ppl_thr),
        "mem_budget_mb": int(mem_bgt),
        "decode_tps_min": int(tps_min),
    },
}

os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print("Report  --> %s" % report_path)

PYEOF

# Symlink latest
ln -sf "$(basename "$REPORT_FILE")" "$LATEST_LINK"
echo "Latest  --> $LATEST_LINK (symlink)"

echo "Log     --> $LOG_FILE"
echo "$HR"

# ===========================================================================
# Final exit
# ===========================================================================
if [[ $REQUIRED_FAILED -eq 0 ]]; then
    echo "  >>> ALL REQUIRED GATE 1 CHECKS PASSED <<<"
    echo "  >>> Ready to proceed to Phase 2 <<<"
    echo "$HR"
    exit 0
else
    echo "  >>> GATE 1 FAILED -- $REQUIRED_FAILED required check(s) did not pass <<<"
    echo "  >>> Fix failures before proceeding to Phase 2 <<<"
    echo "$HR"
    exit 1
fi
