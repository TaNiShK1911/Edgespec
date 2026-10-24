# EdgeSpec-H2O

**Edge-optimized LLM inference with H₂O KV-cache eviction and speculative decoding**

Target: NVIDIA Jetson Nano 4 GB · OPT-1.3B 4-bit (GPTQ) · Python 3.6.9 · PyTorch 1.10.0

## Quick Start (Nano)

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks && sudo nvpmodel -q
source ~/edgespec/venv-edgespec/bin/activate
python tools/check_env.py && bash bench/gate1.sh
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
│  ├─ config.py              # Central configuration
│  ├─ tokenizer.py           # Vendored GPT-2 BPE (OPT vocab/merges)
│  ├─ model/
│  │  ├─ opt.py              # Vendored OPT forward pass
│  │  └─ loader.py           # Low-peak-RAM tensor-by-tensor loader
│  ├─ quant/
│  │  ├─ qlinear.py          # INT4 unpack + dequant reference path
│  │  └─ csrc/               # OPTIONAL fused CUDA dequant-matmul
│  ├─ cache/
│  │  └─ h2o_cache.py        # Phase 2: decayed H₂O cache
│  ├─ spec/                  # Phase 3: speculative decoding
│  └─ utils/
│     ├─ profiler.py         # Perf profiling
│     └─ tegrastats.py       # Jetson tegrastats parser
├─ tools/
│  ├─ check_env.py           # Environment validation
│  ├─ offline_quantize.py    # GPTQ offline quantization
│  ├─ export_golden.py       # Export golden logits & checksums
│  ├─ gen_logs.py            # Synthetic log generator
│  ├─ eval_ppl.py            # WikiText-2 perplexity evaluation
│  └─ eval_rouge.py          # ROUGE-2 evaluation
├─ tests/                    # pytest test suite
├─ bench/
│  ├─ gate1.sh               # Gate 1 runner
│  └─ results/               # Gate output JSONs
├─ data/                     # Vocab, merges, golden files
└─ Makefile
```

## Phases

- **Phase 1** (H0–H8): Foundation — model runs quantized and profiled on Nano
- **Phase 2** (H8–H14): Memory — H₂O KV cache with rollback API
- **Phase 3** (H14–H21): Speed — speculative decoding + telemetry UI + demo
