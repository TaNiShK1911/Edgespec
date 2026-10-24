# -*- coding: utf-8 -*-
"""EdgeSpec-H2O central configuration.

All tunables in one place. Python-3.6 safe (no walrus, no f-string =, no annotations future).
"""

import os


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
MODEL_NAME = "opt-1.3b"
MODEL_VARIANT = "facebook/opt-1.3b"  # HF model id (for offline use only)
SMALL_MODEL_NAME = "opt-125m"
SMALL_MODEL_VARIANT = "facebook/opt-125m"

# OPT architecture constants
OPT_CONFIGS = {
    "opt-125m": {
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 768,
        "d_ffn": 3072,
        "vocab_size": 50272,
        "max_positions": 2048,
        "position_offset": 2,  # OPT uses offset=2 for learned positions
        "activation": "relu",
        "layer_norm_eps": 1e-5,
        "bos_token_id": 2,  # </s>
        "eos_token_id": 2,
        "pad_token_id": 1,
        "tie_word_embeddings": True,
        "pre_layernorm": True,  # OPT-125M uses pre-LN (do_layer_norm_before=True)
    },
    "opt-350m": {
        "n_layers": 24,
        "n_heads": 16,
        "d_model": 1024,
        "d_ffn": 4096,
        "vocab_size": 50272,
        "max_positions": 2048,
        "position_offset": 2,
        "activation": "relu",
        "layer_norm_eps": 1e-5,
        "bos_token_id": 2,
        "eos_token_id": 2,
        "pad_token_id": 1,
        "tie_word_embeddings": True,
        "pre_layernorm": False,  # OPT-350M uses post-LN (do_layer_norm_before=False)
    },
    "opt-1.3b": {
        "n_layers": 24,
        "n_heads": 32,
        "d_model": 2048,
        "d_ffn": 8192,
        "vocab_size": 50272,
        "max_positions": 2048,
        "position_offset": 2,
        "activation": "relu",
        "layer_norm_eps": 1e-5,
        "bos_token_id": 2,
        "eos_token_id": 2,
        "pad_token_id": 1,
        "tie_word_embeddings": True,
        "pre_layernorm": True,
    },
}

# --------------------------------------------------------------------------- #
#  Quantization (GPTQ)
# --------------------------------------------------------------------------- #
GPTQ_BITS = 4
GPTQ_GROUP_SIZE = 128  # g128; -1 = no grouping
GPTQ_SYMMETRIC = False
GPTQ_ACT_ORDER = False  # avoid act-order — needs permutation in kernel
GPTQ_CALIBRATION_NSAMPLES = 128
GPTQ_CALIBRATION_SEQLEN = 2048

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
GOLDEN_DIR = os.path.join(DATA_DIR, "golden")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.json")
MERGES_FILE = os.path.join(DATA_DIR, "merges.txt")

# --------------------------------------------------------------------------- #
#  KV cache (Phase 1: full cache; Phase 2: H2O)
# --------------------------------------------------------------------------- #
MAX_SEQ_LEN = 2048
PREFILL_CHUNK_SIZE = 128  # tokens per chunk in layer-major prefill

# --------------------------------------------------------------------------- #
#  H2O cache (Phase 2)
# --------------------------------------------------------------------------- #
H2O_DEFAULTS = {
    "n_hh": 128,               # heavy-hitter slots per head
    "n_local": 128,            # local (recent) window slots per head
    "n_pin": 4,                # pinned attention-sink positions
    "n_scratch": 8,            # scratch slots for speculative tokens (gamma_max + 1)
    "decay": 0.98,             # score decay lambda (1.0 = plain H2O)
    "evict_every": 1,          # evict after every N commits (1 = every commit)
}

# Budget presets (fraction of 2048 context)
H2O_BUDGET_PRESETS = {
    "10pct": {"n_hh": 78, "n_local": 128, "n_pin": 4},       # ~10% = 210 tok
    "20pct": {"n_hh": 278, "n_local": 128, "n_pin": 4},      # ~20% = 410 tok
    "30pct": {"n_hh": 482, "n_local": 128, "n_pin": 4},      # ~30% = 614 tok
    "50pct": {"n_hh": 892, "n_local": 128, "n_pin": 4},      # ~50% = 1024 tok
    "100pct": {"n_hh": 1916, "n_local": 128, "n_pin": 4},    # 100% = 2048 tok
}

# Decay sweep values for ablation
H2O_DECAY_SWEEP = [1.0, 0.99, 0.98, 0.95]

# Pin sweep values for ablation
H2O_PIN_SWEEP = [0, 1, 4]

# Importance layers (for Phase 3 drafter token-importance vector)
H2O_IMPORTANCE_LAYERS = [8, 16, 23]  # layers to accumulate importance from

# --------------------------------------------------------------------------- #
#  Profiling / benchmarking
# --------------------------------------------------------------------------- #
WARMUP_STEPS = 3
BENCH_CONTEXTS = [128, 512, 1024, 2048]
VERIFY_PROBE_RANGE = list(range(1, 9))  # k = 1..8

# --------------------------------------------------------------------------- #
#  Device
# --------------------------------------------------------------------------- #
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "float16"  # FP16 on both Nano and laptop

# --------------------------------------------------------------------------- #
#  Gate thresholds (Phase 1)
# --------------------------------------------------------------------------- #
GATE1 = {
    "top1_agreement_pct": 98.0,       # G1.3
    "mean_abs_logit_diff": 0.05,      # G1.3
    "layer_matmul_rel_error": 5e-3,   # G1.4
    "ppl_tolerance": 0.05,            # G1.5
    "first64_match_pct": 90.0,        # G1.6
    "peak_ram_gib": 3.3,              # G1.7
    "min_tok_per_sec": 1.0,           # G1.8
}

# --------------------------------------------------------------------------- #
#  Gate thresholds (Phase 2)
# --------------------------------------------------------------------------- #
GATE2 = {
    "equivalence_max_abs_diff": 1e-3,    # G2.2: no-eviction logits == full-cache
    "commit_rollback_max_diff": 1e-3,    # G2.3: commit/rollback correctness
    "mem_alloc_tolerance_pct": 5.0,      # G2.4: within 5% of formula
    "mem_saving_min_mib": 250,           # G2.4: >= 250 MiB lower than full-KV
    "rouge2_pct_of_full": 90.0,          # G2.5: ROUGE-2 >= 90% of full cache
    "delta_ppl_max": 1.0,                # G2.5: ΔPPL <= +1.0 vs full cache
    "ppl_nano_laptop_tolerance": 0.05,   # G2.5: Nano vs laptop ΔPPL <= 0.05
    "scoring_overhead_max_pct": 15.0,    # G2.7: scoring+eviction <= 15% of step
    "soak_rss_drift_max_mib": 20,        # G2.8: RSS drift <= 20 MiB in 10 min
    "soak_duration_sec": 600,            # G2.8: 10-min soak duration
}

# --------------------------------------------------------------------------- #
#  Speculative Decoding (Phase 3)
# --------------------------------------------------------------------------- #
SPEC_DEFAULTS = {
    "gamma_max": 6,            # maximum draft length
    "gamma_fixed": 4,          # fixed gamma when controller is off
    "lenience": 1.0,           # strict/lossless by default
    "mode": "greedy",          # 'greedy' or 'sampling'
    "temperature": 1.0,        # sampling temperature
    "drafter_max_n": 4,        # max n-gram length for suffix matching
    "drafter_selector": "importance",  # 'importance', 'most_recent', 'longest'
    "gamma_ema_alpha": 0.1,    # EMA smoothing for gamma controller
}

# Lenience sweep values for ablation (G3.6)
LENIENCE_SWEEP = [1.0, 0.8, 0.6, 0.45, 0.3]

# --------------------------------------------------------------------------- #
#  Gate thresholds (Phase 3)
# --------------------------------------------------------------------------- #
GATE3 = {
    "sampler_chi2_p_min": 0.01,          # G3.1: chi-square p > 0.01
    "spec_mismatch_pct_max": 1.0,        # G3.2: <= 1% token mismatches
    "logit_margin_near_tie": 0.05,       # G3.2: top-2 margin for near-tie
    "speedup_pass": 1.5,                 # G3.4: minimum speedup for pass
    "speedup_target": 2.0,               # G3.4: target speedup
    "speedup_stretch": 2.8,              # G3.4: stretch goal
    "mem_overhead_max_mib": 50,          # G3.5: Mode D peak <= Phase-2 + 50 MiB
    "soak_duration_sec": 1800,           # G3.5: 30-min soak
    "ui_updates_per_sec_min": 5,         # G3.7: >= 5 updates/s
    "ui_decode_slowdown_pct_max": 5.0,   # G3.7: <= 5% decode slowdown
    "waf_regex_compile_rate": 100.0,     # G3.8: 100% regex compile rate
    "waf_attack_recall_min": 80.0,       # G3.8: >= 80% attack recall
    "waf_fpr_max": 5.0,                  # G3.8: <= 5% FPR on benign
    "triage_accuracy_min": 80.0,         # G3.8: >= 80% top offender accuracy
}

# --------------------------------------------------------------------------- #
#  Demo / telemetry
# --------------------------------------------------------------------------- #
TELEMETRY_PORT = 8765
DEMO_LOG_WINDOW_SIZE = 200              # lines per triage window

