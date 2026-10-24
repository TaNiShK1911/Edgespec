# EdgeSpec-H2O

**Edge-optimized LLM inference with H₂O KV-cache eviction and speculative decoding**

Target: NVIDIA Jetson Nano 4 GB · OPT-1.3B 4-bit (GPTQ) · Python 3.6.9 · PyTorch 1.10.0

## Quick Start (Nano)

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks && sudo nvpmodel -q
source ~/edgespec/venv-edgespec/bin/activate
python tools/check_env.py && bash bench/gate3.sh
```

## Quick Start (Laptop / Dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python tools/check_env.py
pytest tests/ -v
```

## Repo Layout

```
edgespec-h2o/
├─ edgespec/
│  ├─ config.py              # Central configuration (all 3 phases)
│  ├─ tokenizer.py           # Vendored GPT-2 BPE (OPT vocab/merges)
│  ├─ model/
│  │  ├─ opt.py              # Vendored OPT forward pass (full + H2O)
│  │  └─ loader.py           # Low-peak-RAM tensor-by-tensor loader
│  ├─ quant/
│  │  ├─ qlinear.py          # INT4 unpack + dequant reference path
│  │  └─ csrc/               # OPTIONAL fused CUDA dequant-matmul
│  ├─ cache/
│  │  └─ h2o_cache.py        # Phase 2: decayed H₂O cache + commit/rollback
│  ├─ spec/                  # Phase 3: speculative decoding
│  │  ├─ drafter.py          # H₂O-weighted suffix-match drafter (zero-RAM)
│  │  ├─ sampler.py          # Speculative acceptance sampler (greedy/sampling)
│  │  ├─ verifier.py         # Full verify loop (drafter + model + cache)
│  │  └─ gamma_ctrl.py       # Adaptive γ controller using measured V(k)
│  ├─ telemetry/
│  │  └─ server.py           # SSE telemetry server + browser dashboard
│  └─ utils/
│     ├─ profiler.py         # Perf profiling + verify-cost probe
│     └─ tegrastats.py       # Jetson tegrastats parser
├─ tools/
│  ├─ check_env.py           # Environment validation
│  ├─ offline_quantize.py    # GPTQ offline quantization
│  ├─ export_golden.py       # Export golden logits & checksums
│  ├─ gen_logs.py            # Synthetic log generator (Akamai demo)
│  ├─ demo_akamai.py         # Demo prompts, WAF validator, evaluator
│  ├─ bench_matrix.py        # 4-mode benchmark matrix (A/B/C/D)
│  ├─ eval_ppl.py            # WikiText-2 perplexity evaluation
│  └─ eval_rouge.py          # ROUGE-2 evaluation
├─ tests/
│  ├─ test_tokenizer.py
│  ├─ test_qlinear.py
│  ├─ test_model.py
│  ├─ test_h2o_cache.py
│  └─ test_spec.py           # Phase 3 tests (29 tests)
├─ bench/
│  ├─ gate1.sh               # Gate 1 runner (Foundation)
│  ├─ gate2.sh               # Gate 2 runner (H₂O cache)
│  └─ gate3.sh               # Gate 3 runner (Speculation + demo)
├─ data/                     # Vocab, merges, golden files, synthetic logs
├─ results/                  # Gate outputs, benchmark CSVs
└─ Makefile
```

## Phases

- **Phase 1** (H0–H8): Foundation — model runs quantized and profiled on Nano
- **Phase 2** (H8–H14): Memory — H₂O KV cache with rollback API
- **Phase 3** (H14–H21): Speed — speculative decoding + telemetry UI + Akamai demo

## Phase 3 — Speculative Decoding

### Architecture

```
 prompt ──► prefill_h2o ──► H₂O cache (compressed)
                              │
                              ▼
           ┌──── speculative decode loop ─────────────┐
           │  SuffixDrafter: n-gram match over token   │
           │    history, weighted by H₂O importance    │
           │    → γ draft tokens (zero RAM!)           │
           │                                           │
           │  Verifier: ONE forward_block_h2o over     │
           │    [last_tok + γ drafts]                  │
           │                                           │
           │  SpeculativeSampler: accept/reject per    │
           │    Alg. 1 (greedy/sampling, lenience l)   │
           │                                           │
           │  GammaController: adaptive γ from         │
           │    per-position EMA + measured V(k)       │
           │                                           │
           │  commit → update_scores → evict → telem   │
           └───────────────────────────────────────────┘
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| **Drafter** | `edgespec/spec/drafter.py` | Zero-RAM suffix-match drafter using n-gram index + H₂O importance weighting. 3 selectors: `importance`, `most_recent`, `longest` |
| **Sampler** | `edgespec/spec/sampler.py` | Speculative acceptance (Leviathan et al. Alg. 1 + App. A.5). Greedy and sampling modes with lenience parameter |
| **Verifier** | `edgespec/spec/verifier.py` | Full decode loop tying drafter, model, H₂O cache, and sampler together |
| **γ Controller** | `edgespec/spec/gamma_ctrl.py` | Adaptive γ ∈ {0..6} maximizing E[tokens]/(V(γ+1) + γ·c) using measured V(k) |
| **Telemetry** | `edgespec/telemetry/server.py` | SSE server + browser dashboard with live tok/s, α̂, H₂O heatmap, token stream, control sliders |
| **Demo** | `tools/demo_akamai.py` | Few-shot prompts for log triage + WAF rule drafting, with regex validator and ground-truth evaluator |
| **Benchmark** | `tools/bench_matrix.py` | 4-mode matrix (A: baseline, B: H₂O, C: spec, D: EdgeSpec-H₂O) with auto speedup reporting |

### Benchmark Modes

| Mode | Weights | KV Cache | Speculation | Description |
|------|---------|----------|-------------|-------------|
| A | 4-bit | Full | Off | Baseline |
| B | 4-bit | H₂O | Off | Memory savings only |
| C | 4-bit | Full | On | Speculation only |
| **D** | 4-bit | H₂O | **On** | **Full EdgeSpec-H₂O** |

### Running

```bash
# Unit tests (no GPU needed)
make test-spec

# Gate 3 — laptop mode
make gate3-laptop

# Gate 3 — full Nano run
make gate3-full

# Benchmark matrix
make bench-matrix

# Akamai demo prompts
make demo

# Live telemetry dashboard (on Nano, view in laptop browser)
python -c "from edgespec.telemetry import start_server; start_server(); input()"
# Open http://<nano-ip>:8765 in browser
```

### Gate 3 Thresholds

| Check | Criterion | Level |
|-------|-----------|-------|
| G3.1 | Sampler chi-square p > 0.01; lenience monotonic | MUST |
| G3.2 | Spec vs non-spec ≤ 1% mismatch (greedy, no eviction, l=1) | MUST |
| G3.3 | Mode D quality = Mode B at l=1 | MUST |
| G3.4 | Speedup ≥ 1.5× (pass) / 2.0× (target) / 2.8× (stretch) | MUST |
| G3.5 | Peak RAM ≤ Phase-2 + 50 MiB; 30-min soak | MUST |
| G3.6 | Lenience curve for l ∈ {1.0, 0.8, 0.6, 0.45, 0.3} | MUST |
| G3.7 | UI ≥ 5 updates/s, ≤ 5% decode slowdown | MUST |
| G3.8 | WAF regex compile 100%; attack recall ≥ 80% | SHOULD |
| G3.9 | Tag, README, results CSVs, deck numbers | MUST |
