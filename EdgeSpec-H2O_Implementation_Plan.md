# EdgeSpec-H2O — Implementation Plan (3 gated phases)

**Hashcode 14.0 · Track 01 (LLM Inference) · Team Code Obsessed · Target: NVIDIA Jetson Nano 4GB**

> **Rule of the build:** a phase is *done* only when its **Gate** passes **on the Jetson Nano** (not the laptop). A failed MUST item blocks the next phase. Every gate is a script that writes `results/gateN.json` + logs, and a passing run is tagged in git (`phase1-pass`, `phase2-pass`, `phase3-pass`).

**Naming:** your readiness report's "Phases 1–3" (inventory, memory, software) are called **R1–R3** here. This plan's Phases 1–3 are the *build* phases.

---

## 0. Inputs this plan is built on

| Source | What was used |
|---|---|
| Speculative Decoding (Leviathan et al., arXiv 2211.17192) | Alg. 1, Eq. (1), Thm 3.8, Cor 3.9, §3.5 (choosing γ), §3.6 (n-gram / copy drafters), Table 3, Appendix A.5 (lenience) |
| H₂O (Zhang et al., arXiv 2306.14048) | §3.2 heavy hitters, Alg. 1, Appendix A (circular-queue implementation), Appendix B.2 (accumulated-score bias), Table 2, Fig. 4 |
| GPTQ (Frantar et al., arXiv 2210.17323) | §4 algorithm, Table 3 (OPT perplexities), Fig. 4 (grouping), §5 (kernels), §6 limitations |
| Lexico (Kim et al., arXiv 2412.08890) | §3.4, Table 7 (OMP latency), §4.2.4; used as a *complementary* idea, not built (see §3) |
| Deck `Hashcode14_EdgeSpec-H2O_Filled.pptx` | Your design, targets, memory budget, milestones |
| `EdgeSpec-H2O_Readiness_Report_Phases_1-3.docx` | Jetson config and the Transformers/OPT blocker |
| Hashcode 14.0 page | 24 h, teams of 2–3, Track 01 includes H₂O and Lexico flagged "Sponsored by Akamai" |

**Jetson configuration taken from the readiness report** (tell me if any of it is outdated): Nano 4GB (Tegra X1, Maxwell, SM 5.3, 128 CUDA cores) · JetPack 4.6.6 / L4T R32.7.6 · Ubuntu 18.04 · **Python 3.6.9** · **PyTorch 1.10.0 (NVIDIA build, CUDA 10.2)** · 3.9 GB RAM (~577 MB used at baseline) · ~1.97 GB ZRAM, no disk swap · venv `~/edgespec/venv-edgespec` · Transformers 4.18.0 has **no `OPTForCausalLM`** · GUI→headless switch dropped USB networking, so the default boot target is unchanged.

---

## 1. Reality check: deck claims vs. what the papers support

These need fixing *before* they become gate thresholds or slide claims. Nothing here kills the idea; it makes the targets defensible.

| # | Deck claim | What the papers / arithmetic say | Action in this plan |
|---|---|---|---|
| 1 | "<0.3 perplexity delta" for the whole system | GPTQ Table 3: OPT-1.3B WikiText-2 PPL **14.63 → 15.47 (+0.84)** at 4-bit; OPT-125M 27.65 → 31.12. Grouping (g128) narrows the gap (GPTQ Fig. 4). H₂O adds more on top. | Report ΔPPL separately per stage (GPTQ, H₂O, spec). Spec decoding at l=1 adds **0** by construction. Use g128. |
| 2 | "~700–750 MB" 4-bit weights | GPTQ keeps embeddings/output layer in FP16. OPT-1.3B: 1.208 B layer params × 0.5 B ≈ 576 MiB + scales/zeros (g128) ≈ 36 MiB + FP16 embeddings (50272×2048×2 B) ≈ 196 MiB + positions ≈ 8 MiB → **≈ 0.8 GiB** | Use 0.8 GiB in the budget; measure in G1.7 |
| 3 | "45 MB KV at 20% budget" | OPT-1.3B KV = 2×24×2048×2 B = **192 KiB/token**. 256 tokens (128 HH + 128 local) = **48 MiB = 12.5%** of 2048. True 20% (410 tok) = **77 MiB**. Full 2048 = **384 MiB**. | Quote both numbers; sweep budgets 10–50% |
| 4 | "2,048+ token context" | OPT has learned absolute positions for 2048 tokens. H₂O's 4M-token result used StreamingLLM-style position rolling, which doesn't transfer to OPT as-is. | Target = **full 2048-token context**; "beyond 2048" is out of scope |
| 5 | α ≥ 0.55 on summarization | Speculative-decoding paper only *measured* n-gram drafters (α ≈ 0.05–0.23, Table 3); context-copy drafters are *suggested* (§3.6), not measured. Abstractive XSUM may be low; logs / RAG / few-shot should be high. | Treat as hypothesis; measure per workload; lead the demo with logs |
| 6 | 2.0–2.8× speedup | Paper's theory assumes γ+1 verifications cost the same as 1 (§3.3). On a 128-core GPU that may not hold. See §1.1 table. | Phase 1 **measures verify cost V(k)** and sets the achievable target |
| 7 | "Lossless at l = 1.0" | True *w.r.t. the target model as configured* (Appendix A.1). But **H₂O eviction itself is lossy vs. the full cache**; spec adds nothing on top. Also FP16 batch-shape noise can flip near-ties in greedy decoding. | Test spec correctness with eviction **off**; test eviction by quality metrics |
| 8 | GPTQ run on the Nano | GPTQ quantization needs the FP16 model + Hessians; the paper used an A100. Paper's fast kernels target A100/A6000; **no kernel exists for SM 5.3**. GPTQ saves memory traffic, not FLOPs (§6). | Quantize **offline** (laptop/Colab); write a dequant path (+ optional kernel) for the Nano |
| 9 | Decay fixes early-token bias | H₂O App. B.2 reports the bias but its averaged-score fix *degraded* quality. Decay is a plausible but **untested** alternative. Also H₂ often sit at sentence starts. | Ablate λ ∈ {1.0, 0.99, 0.98, 0.95}; add `pin_first_n`; report honestly even if λ=1 wins |
| 10 | 20% budget "matches full cache" | H₂O's evidence is OPT-30B/66B, LLaMA-7B–30B, GPT-NeoX-20B. **Nothing at 1.3B.** | Laptop sweep early in Phase 2 *before* committing the demo |
| 11 | TinyLlama-1.1B as 2nd model | It uses GQA (4 KV heads): KV ≈ **22 KiB/token (44 MiB at 2048)** — little for H₂O to save, and Lexico's paper notes eviction methods fit poorly with GQA. | **OPT-1.3B (MHA) primary.** TinyLlama = stretch only |
| 12 | OPT-1.3B "summarizes logs / writes WAF rules" | OPT-1.3B is a **base** LM, not instruction-tuned. | Few-shot prompt templates + automatic validator (§7, Phase 3) |
| 13 | "Akamai edge-server access logs" | No real Akamai logs are available to the team. | Generate **synthetic, labelled** logs; label them synthetic everywhere |

### 1.1 What speedup is realistic? (computed from the paper's Eq. 1, drafter cost c≈0)

Speedup = E[tokens] / (1 + β·γ), where E[tokens] = (1−α^(γ+1))/(1−α) and **β = marginal cost of each extra verified token, as a fraction of a 1-token step**. Best γ ∈ 1..6 shown. β is measured in Phase 1 (G1.9).

| α \ β | 0 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|
| 0.5 | 1.98× | 1.63× | 1.46× | 1.25× |
| 0.6 | 2.43× | 1.92× | 1.67× | 1.40× |
| 0.7 | 3.06× | 2.35× | 1.98× | 1.58× |
| 0.8 | 3.95× | 3.04× | 2.47× | 1.87× |

The deck's 2.0–2.8× needs **β ≲ 0.05 and α ≳ 0.6–0.7**. Log/RAG workloads can plausibly hit that; XSUM probably won't. Gate 3 therefore uses *Pass 1.5× / Target 2.0× / Stretch 2.8×*.

---

## 2. Jetson Nano compatibility consequences

| Constraint (from readiness report) | Consequence for the code |
|---|---|
| Python 3.6.9 | No `dataclasses` stdlib (use plain classes/namedtuples or `pip install dataclasses`), no `from __future__ import annotations`, no walrus, no `breakpoint()`. CI: `python3.6 -m py_compile` on every file. NumPy ≤ 1.19.5. |
| PyTorch 1.10.0 | No `scaled_dot_product_attention`, no `torch.compile`, no bf16 on Maxwell. Write attention by hand (H₂O needs the probabilities anyway). `torch.cuda.CUDAGraph` exists (beta) — optional optimization. |
| SM 5.3 | Native FP16 math, **no tensor cores, no `dp4a`** → INT4 weights must be dequantized to FP16 (in registers if a kernel is written). |
| Transformers OPT blocked | **No `transformers` in the inference path.** Vendored pure-PyTorch OPT; own GPT-2-BPE tokenizer (the original OpenAI `encoder.py` logic + `regex`) using OPT's `vocab.json`/`merges.txt`, because `tokenizers` (Rust) may have no Py3.6/aarch64 wheel. Fallback: tokenize on the laptop. |
| 3.9 GB unified RAM | CPU and GPU share it: loading must be **tensor-by-tensor** from a custom file (never `torch.load` a 2.6 GB checkpoint). Prefill must be **layer-major + chunked** (a naive 2048-token prefill materializes 32×2048×2048 FP16 scores = 268 MB per layer). |
| ZRAM only, no disk swap | Keep it that way. Benchmarks are invalid if the system swaps; gates check `vmstat` si/so. Do **not** add the 8 GB SD-card swap (slow, wears the card, hides regressions). |
| Headless switch broke USB net | Don't retry it. SSH over Wi-Fi/Ethernet, run the telemetry server on the Nano, **render the UI in the laptop's browser** (a browser on the Nano would eat ~0.4 GB+). Revisit headless only if G1.7 shows memory is binding. |
| Max-N 10 W | `sudo nvpmodel -m 0 && sudo jetson_clocks`; use a fan and a proper 5 V 4 A barrel supply; log temps. |

---

## 3. Architecture

### 3.1 Data flow

```
 OFFLINE (laptop/Colab GPU)                         ON NANO
 ───────────────────────────                        ──────────────────────────────────────────
 OPT-1.3B FP16 ──GPTQ 4-bit g128──► packed export ─► loader (tensor-by-tensor, low peak RAM)
 (C4: 128×2048 calib, per paper §5)  + golden logits   │
                                                       ▼
 prompt ──► BPE tokenizer ──► layer-major chunked prefill ──► H2OCache (budget, decay λ, pins)
                                                       │              ▲  commit / rollback API
                                                       ▼              │
                          ┌──────────── decode loop ─────────────────┴────────────┐
                          │  drafter (suffix-match over token history,            │
                          │           weighted by H2O importance)  → γ drafts     │
                          │  verifier: ONE forward over [last tok + γ drafts]     │
                          │  speculative sampler (greedy / sampling, lenience l)  │
                          │  commit n+1 tokens → update scores → evict → telemetry│
                          └───────────────────────────────────────────────────────┘
```

### 3.2 Repo layout (Python-3.6-safe)

```
edgespec-h2o/
├─ README.md   requirements-nano.txt (pinned)   Makefile (gate1/gate2/gate3)
├─ edgespec/
│  ├─ config.py
│  ├─ tokenizer.py            # vendored GPT-2 BPE, OPT vocab/merges
│  ├─ model/opt.py            # vendored OPT forward, static buffers, forward_block()
│  ├─ model/loader.py         # low-peak-RAM loader
│  ├─ quant/qlinear.py        # int4 unpack+dequant reference path
│  ├─ quant/csrc/             # OPTIONAL fused dequant-matmul (time-boxed)
│  ├─ cache/h2o_cache.py      # decayed H2O cache, scratch slots, commit/rollback
│  ├─ spec/drafter.py  verifier.py  sampler.py  gamma_ctrl.py
│  ├─ telemetry/server.py  static/   # SSE stream + browser UI
│  └─ utils/profiler.py  tegrastats.py
├─ tools/  offline_quantize.py  export_golden.py  gen_logs.py  eval_ppl.py  eval_rouge.py
├─ tests/  (pytest-compatible, no GPU-only asserts unless marked)
├─ bench/  gate1.sh gate2.sh gate3.sh   results/   docs/
```

Development rule: **all algorithmic code runs on the laptop too** (same files), so sweeps/ablations run fast there and only the *confirming* runs happen on the Nano.

### 3.3 Where Lexico fits (it is Akamai-sponsored, so expect a judge question)

- Lexico compresses **each retained KV vector** via sparse coding (OMP over a 4096-atom universal dictionary); H₂O **drops tokens**. They act on different axes and could multiply; Lexico's related-work section itself says eviction and Lexico may work well together.
- **Not built on the Nano:** OMP costs 26–40 ms/token on the paper's hardware (Table 7) and needs a trained dictionary (~2 h on an A100). A 128-core Maxwell GPU can't afford that per token.
- **Pitch line:** "H₂O shrinks *how many* tokens we keep; Lexico-style coding would shrink *how big* each one is — future work once an OMP kernel fits an edge GPU."

### 3.4 Akamai alignment (what to emphasize)

From Akamai's public positioning (Inference Cloud launch press release: https://www.akamai.com/newsroom/press-release/akamai-inference-cloud-transforms-ai-from-core-to-edge-with-nvidia ; Cloud Inference launch: https://www.akamai.com/newsroom/press-release/akamai-sharpens-its-ai-edge-with-launch-of-akamai-cloud-inference):

1. **Inference moves to the edge** for latency and cost → our story is *inference on a 4 GB edge box*.
2. **Security is their largest business** (WAF, DDoS, bot management) → our demo is **WAF-rule drafting and log triage**.
3. **Privacy/locality** → logs never leave the device; demo with network egress disabled.
4. Sponsored papers in the track are **H₂O** and **Lexico** → H₂O is the core of this project.

Do **not** claim Akamai uses this approach or hardware; claim only that it targets the same constraint (small, local, secure inference).

---

## 4. Memory model (replace slide 9 with *measured* values after G1.7)

| Component | Deck | Expected (computed) | Notes |
|---|---|---|---|
| OS + CUDA context + PyTorch | 1,100 MB | **1.1–1.6 GB (measure)** | PyTorch CUDA init on Nano is often the biggest unknown |
| 4-bit weights (OPT-1.3B g128 + FP16 embeds) | 750 MB | **≈ 820 MiB** | §1 row 2 |
| KV cache, full 2048 ctx | (OOM) | 384 MiB | baseline for Phase 1 |
| KV cache, H₂O 256 tok / 410 tok (20%) | 45 MB | 48 MiB / 77 MiB | + scratch for γ_max+1 slots ≈ 1.3 MiB |
| Activations / logits / temp dequant buffers | not listed | 100–300 MiB | dequant temp for 2048×8192 FP16 ≈ 32 MiB per matrix |
| Drafter + token index | 2 MB | < 5 MB | token ids only |
| **Expected total (H₂O on)** | 1,897 MB | **≈ 2.3–3.0 GiB** | Gate cap 3.3 GiB |

---

# PHASE 1 — Foundation: the model runs, quantized and profiled, on the Nano

**Goal:** greedy generation from 4-bit OPT-1.3B on the Nano with a vendored PyTorch model, a baseline measured to the decimal, and the *go/no-go measurement for speculative decoding* (verify cost). No H₂O, no speculation yet.
**Time:** H0–H8 (gate run H7–H8) · **Timebox for fixes after a failed gate: 90 min, then use the fallback ladder (§8).**

### 1.1 Tasks

| ID | Task | Owner (deck roles) | Notes |
|---|---|---|---|
| P1-1 | Repo, pinned `requirements-nano.txt`, `check_env.py`, `nvpmodel` Max-N + `jetson_clocks`, `tegrastats --interval 500 --logfile` wrapper | M1 | Keep existing `venv-edgespec`; install NumPy ≤ 1.19.5 (clears the NumPy warning in the report) |
| P1-2 | Offline GPTQ (laptop/Colab): OPT-125M and OPT-1.3B; **g128 and no-group** variants; C4 128×2048 calibration (paper §5); **avoid `act-order`** (needs permutation in the kernel). Reference code: github.com/IST-DASLab/gptq | M1 | Record WikiText-2 PPL FP16 vs 4-bit; compare to GPTQ Table 3 (14.63 vs 15.47 for 1.3B) |
| P1-3 | Export format: per-tensor offsets in one binary file — `qweight` uint8 (2×int4/byte), `scales` fp16, `zeros`, FP16 embeddings/LN/bias; plus **golden files**: token-id fixtures, FP32 logits for 5 prompts, per-layer weight checksums | M1 | Loader reads one tensor at a time → low peak RAM |
| P1-4 | Vendored OPT forward: pre-LN, learned positions with offset 2, ReLU FFN, tied LM head, BOS `</s>` (id 2). `forward_block(tokens, cache)` accepting **k ≥ 1 new tokens** (needed in Phase 3). Static preallocated KV buffers. Manual attention (optionally returns probabilities) | M2 | Validate on **OPT-125M FP16 first** (fits easily), then 1.3B |
| P1-5 | Layer-major, chunked prefill (all prompt tokens through layer *l*, then *l+1*; attention in 128-token chunks) | M2 | Peak transient = one layer's KV (~16 MiB) + one score chunk, not 268 MB/layer |
| P1-6 | `QLinear`: unpack int4 → dequant → FP16 matmul (reference path). Pack/unpack unit test | M1 | Dequant cost is independent of k tokens → favorable for speculation |
| P1-7 | Tokenizer (vendored BPE) with golden-ID test; fallback = laptop-side tokenization | M3 | |
| P1-8 | Profiler: prefill vs decode tok/s, per-op GPU time vs CPU launch time, RSS + tegrastats RAM, zram, temps | M3 | A57 @1.48 GHz may make Python/launch overhead the bottleneck, not bandwidth |
| P1-9 | **Verify-cost probe**: time one forward over k = 1…8 new tokens at ctx 512 → V(k), β | M3 | The single most important Phase-1 number |
| P1-10 | *Time-boxed (≤ 2 h)* fused CUDA dequant-matmul for small M (≤ 8 rows), `sm_53`, built via `torch.utils.cpp_extension` (needs `ninja`) | M1 | If not passing by the box, **drop it**; reference path stays |
| P1-11 | Eval harness (WikiText-2 PPL, XSUM/CNN-DM ROUGE-2) usable on laptop and Nano | M3 | Needed in Phase 2 |
| P1-12 | `gen_logs.py` synthetic log generator (see Phase 3 §7.2) — data only, no model dependency | M3 | Allowed as parallel prep |

### 1.2 Commands to run on the Nano (reference)

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks && sudo nvpmodel -q
tegrastats --interval 500 --logfile results/tegrastats_p1.log &
free -m; cat /proc/swaps; vmstat 1 5          # confirm no swap activity during runs
source ~/edgespec/venv-edgespec/bin/activate
python tools/check_env.py && bash bench/gate1.sh
```

### 1.3 GATE 1 — must all pass on the Nano

| ID | Check | Pass criterion | Level |
|---|---|---|---|
| G1.1 | Environment | `check_env.py` green: Py 3.6.9, torch 1.10.0, CUDA available, NumPy imports cleanly, `nvcc` found, Max-N active; `grep -r transformers edgespec/` finds nothing | MUST |
| G1.2 | Tokenizer | Token IDs **exactly equal** to HF golden for ≥ 50 strings (unicode, newlines, code, logs) | MUST |
| G1.3 | Numerics, OPT-125M FP16 | vs laptop FP32 golden logits: top-1 agreement ≥ 98 % over ≥ 500 positions; mean abs logit diff ≤ 0.05 | MUST |
| G1.4 | 4-bit pack/unpack | Dequantized weights **bit-exact** vs offline reference for *every* layer (checksums); layer matmul rel. error ≤ 5e-3 vs FP32 reference | MUST |
| G1.5 | Quality on Nano | OPT-1.3B 4-bit WikiText-2 PPL (first 64 windows × 1024 tok) within **±0.05** of the laptop value for the same checkpoint; ΔPPL vs FP16 recorded | MUST |
| G1.6 | Generation | 3 prompts × 128 greedy tokens: first-64-token match ≥ 90 % vs laptop 4-bit reference; text coherent on inspection | MUST |
| G1.7 | Memory | 2048-token prompt + 128 new tokens, **full KV**: no OOM, peak tegrastats RAM **≤ 3.3 GiB**, `vmstat` si/so ≈ 0. Output: measured memory table (replaces slide 9) | MUST |
| G1.8 | Baseline throughput | tok/s at ctx 128/512/1024/2048 for 125M FP16 and 1.3B 4-bit; GPU-vs-CPU time breakdown. **1.3B 4-bit ≥ 1.0 tok/s at ctx 512** | MUST |
| G1.9 | Verify-cost probe | V(1..8) table and β = (V(8)−1)/7 recorded. **Decision rule below** | MUST (record) |
| G1.10 | Thermal soak | 10-min decode loop, no crash, no throttle flags; max CPU/GPU temps logged | MUST |
| G1.11 | Fused kernel | Parity with reference path and ≥ 1.3× faster than the dequant path | SHOULD |

**G1.9 decision rule (sets Phase 3's ambition):**
β ≤ 0.10 → speculation is clearly viable, aim for the 2.0× target · 0.10 < β ≤ 0.25 → viable with γ ≤ 3, aim ≥ 1.5× · β > 0.25 → fix the verify path first (kernel / fewer Python ops / CUDA graphs); if still > 0.25, reframe the headline as **H₂O memory + throughput at long context**, and keep speculation as a measured secondary result.

---

# PHASE 2 — Memory: time-decayed H₂O KV cache (with a rollback-safe API)

**Start only after Gate 1 passes.** **Goal:** a bounded KV cache whose memory is flat in context length, with quality measured honestly at 1.3B scale, and an API that Phase 3 can use for speculative commit/rollback.
**Time:** H8–H14 · fix-timebox 90 min.

### 2.1 Design decisions (read before coding)

1. **Per-head eviction** (H₂O Appendix A): each head keeps its own heavy-hitter slots. OPT-1.3B is MHA, so no GQA group-sharing complications.
2. **Layout** per layer: `K,V: [H, cap, d]` FP16 with `cap = n_pin + n_hh + n_local + scratch(γ_max+1)`; `pos: [H, cap]` int16; `score: [H, cap]` FP32. Local window = last `n_local` positions (shared across heads); heavy-hitter slots hold per-head tokens. A token leaving the local window competes against the lowest-score heavy hitter in that head (matches the paper's circular-queue description).
3. **Decay:** `score ← λ·score + attn` applied **sequentially per committed token** (closed form for a block of m rows: weight row *i* by `λ^(m−1−i)`). Defaults: λ = 0.98 from the deck, but λ is *swept*, with λ = 1.0 (plain H₂O) as the control.
4. **Pins:** `pin_first_n` (default 1–4) protect attention-sink tokens; included in the ablation.
5. **Rollback-safety (critical for Phase 3):** eviction is irreversible, so **never evict while a speculative block is pending.** Draft tokens write K/V into *scratch slots*; after acceptance, `commit(n_keep)` keeps the first n_keep slots and discards the rest; **scores update only from committed rows**; eviction runs *after* commit (optionally every `evict_every` steps using the slack slots to amortize cost).
6. **Prefill** is full-attention per the paper (prompt phase), then compressed to budget layer-by-layer in the layer-major loop, so peak transient KV stays at one layer.
7. **Token-importance vector** (for the Phase 3 drafter): `importance[pos]` accumulated from probabilities of a few layers (`importance_layers`, configurable) via one `index_add_` per layer — tokens already evicted keep their last value and decay.

### 2.2 API (what Phase 3 will call)

```python
cache.begin_block(n_new)          # reserve scratch slots
model.forward_block(tokens, cache)# writes K/V to scratch; returns logits [n_new,V] and per-row attn probs
cache.commit(n_keep)              # keep first n_keep rows, free the rest
cache.update_scores(probs, n_keep)# decayed accumulation, sequential-equivalent
cache.evict_to_budget()           # per head: drop lowest-score unprotected slots
cache.stats()                     # bytes, occupancy, score histogram (for UI heatmap)
```

### 2.3 Tasks

| ID | Task | Owner |
|---|---|---|
| P2-1 | `H2OCache` data structure + the API above; configurable `n_hh, n_local, λ, pin_first_n, evict_every` | M2 |
| P2-2 | Attention over cache slots (valid-slot mask; causal mask only inside the new block); FP32 softmax | M2 |
| P2-3 | H₂O-aware layer-major prefill with on-the-fly compression | M2 |
| P2-4 | Baselines in the same harness: **Full**, **Local (recent-only)**, **H₂O λ=1**, **H₂O λ<1** | M3 |
| P2-5 | **Laptop sweep first** (budgets 10/20/30/50/100 %; λ ∈ {1, 0.99, 0.98, 0.95}; pin 0/1/4; WikiText-2 PPL, XSUM ROUGE-2 n=100, CNN/DM n=50) | M3 |
| P2-6 | **Recalibration checkpoint (≈ 90 min in):** read the laptop sweep; adjust G2.5 thresholds and default λ/budget *before* Nano runs | all |
| P2-7 | Nano confirmation runs: 3 budget points, 20 XSUM samples, 2 PPL windows | M1 |
| P2-8 | Unit tests (G2.1–G2.3), soak test, memory curve (KV bytes vs context) | M2/M1 |
| P2-9 | Telemetry hooks: `cache.stats()` → JSON (UI built in Phase 3) | M3 |

### 2.4 GATE 2 — must all pass on the Nano

| ID | Check | Pass criterion | Level |
|---|---|---|---|
| G2.1 | Invariants (unit tests) | occupancy never exceeds `cap`; local window always present; pinned tokens never evicted; evicted slot = per-head argmin of unprotected scores; decay math equals a NumPy reference to 1e-6; positions consistent | MUST |
| G2.2 | Equivalence | With budget ≥ context (no eviction) logits equal the Phase-1 full-cache path (max abs diff ≤ 1e-3) | MUST |
| G2.3 | Commit/rollback | Random accept/reject patterns with eviction off ⇒ final K/V equal a sequential token-by-token feed (≤ 1e-3) | MUST |
| G2.4 | Memory | KV bytes flat across context lengths; allocation within 5 % of the formula; peak RAM at 2048 ctx **≥ 250 MiB lower** than Phase-1 full-KV peak at ≤ 20 % budget | MUST |
| G2.5 | Quality (provisional, recalibrated at P2-6) | H₂O beats Local at equal budget on PPL and ROUGE-2; at 20 % budget: **ROUGE-2 ≥ 90 % of full cache** and **ΔPPL ≤ +1.0** vs full cache; Nano numbers match laptop within fp16 noise (ΔPPL ≤ 0.05) | MUST |
| G2.6 | Decay ablation | λ-sweep table committed; default λ chosen by the data. If λ=1.0 wins, ship λ=1.0 and report the negative result | MUST (report) |
| G2.7 | Speed | Decode tok/s at ctx 2048 with H₂O ≥ Phase-1 full-cache tok/s; scoring+eviction overhead ≤ 15 % of step time | MUST |
| G2.8 | Soak | 10-min continuous generation: RSS drift ≤ 20 MiB, no OOM, temps logged | MUST |
| G2.9 | Budget fallback | If 20 % fails G2.5, find the smallest budget that passes (30/40/50 %) and carry *that* forward | MUST (decide) |

---

# PHASE 3 — Speed + product: H₂O-weighted drafter, adaptive γ/lenience, telemetry UI, Akamai demo

**Start only after Gate 2 passes.** **Goal:** the full EdgeSpec-H2O engine, measured end-to-end against baselines, with a demo that tells the Akamai story.
**Time:** H14–H21, then H21–H24 freeze · fix-timebox 90 min.

### 3.1 Drafter (the "zero-RAM" part)

- **Token history** (all committed token IDs, ≤ 2048 ints) + incremental n-gram index for k = 1…4. Only IDs are stored, so drafting needs **no KV and no weights**; the ~MB-scale cost is the "c ≈ 0" claim.
- **Candidate selection:** find prior positions whose preceding k tokens match the current suffix (back off k = 4→3→2→1); each candidate continuation is **weighted by `importance[pos]`** (the decayed H₂O score of the continuation tokens). Ablate against *most-recent-match* and *longest-match* (these two ≈ prompt-lookup baselines — the honest comparison for the novelty claim).
- **Greedy mode (default for the demo):** draft = highest-weight continuation, one-hot q.
- **Sampling mode:** q(x) = normalized weight mass per next-token id; **use that exact q in the accept/reject rule**.
- **No match ⇒ γ = 0:** do a normal 1-token step (no wasted verification).

### 3.2 Verifier and acceptance rule (paper Alg. 1 + App. A.5)

One forward over `[last committed token] + γ drafts` using `begin_block(γ+1)` and `forward_block`.

- **Sampling:** for draft x̂ with draft prob q and target prob p: accept if `l·q ≤ p`, else accept with probability `p/(l·q)`. On reject, resample from `norm(max(0, p − l·q))`. If all γ accepted, sample a bonus token from p_{γ+1}. With a one-hot drafter this reduces to *accept w.p. min(1, p(x̂)/l)*.
- **Greedy:** accept iff `x̂ = argmax p`; lenience variant (paper footnote 7 — lenience for argmax is defined *before* standardization): accept if `p(x̂) ≥ l·max(p)`.
- Then `commit(n+1)`, `update_scores`, `evict_to_budget`.
- **l = 1 is the strict/lossless setting**; l < 1 trades fidelity for acceptance (no token can be sampled with more than p/l of its true probability — App. A.5).

### 3.3 Dynamic γ controller (extends paper §3.5)

Keep an EMA of acceptance **per draft position** α̂₁…α̂₆ (and per match-length bucket). Expected tokens per step without the i.i.d. assumption:
`E(γ) = 1 + Σ_{j=1..γ} Π_{i≤j} α̂_i`. Choose γ ∈ {0,…,6} maximizing `E(γ) / (V(γ+1) + γ·c)` using the **measured V(k) table from G1.9** (not the paper's flat-cost assumption). The paper notes (§3.5) that varying γ can add up to ~60 % over fixed γ under idealized assumptions — this is where that idea becomes concrete.

### 3.4 Tasks

| ID | Task | Owner |
|---|---|---|
| P3-1 | Token history + n-gram index + importance-weighted candidate selection (+ 2 baseline selectors) | M3 |
| P3-2 | Speculative sampler (greedy + sampling + lenience) with a standalone toy-distribution test harness | M3 |
| P3-3 | Verifier loop on `forward_block` + commit/rollback integration | M2/M3 |
| P3-4 | γ controller using measured V(k); γ=0 fallback; knobs exposed | M3 |
| P3-5 | Telemetry server (SSE; Flask 2.0.x is the last line supporting Py3.6, or a stdlib `ThreadingMixIn` server) + browser UI on the laptop: tok/s, RAM, KV bytes, α̂, γ, **H₂O heatmap**, tokens coloured **green = accepted / blue = corrected**, sliders for KV budget, λ, l | M3 |
| P3-6 | Akamai demo data + prompts + validator (§3.5) | M1/M3 |
| P3-7 | Full benchmark matrix (§3.6) → `results/*.csv` + auto-generated figures | M1 |
| P3-8 | Update deck slides 3, 7, 9 with **measured** numbers; write README | all |
| P3-9 | 2-minute demo script, 3 dry runs, **pre-recorded fallback video** | all |

### 3.5 Akamai demo (synthetic, labelled, offline)

- **Synthetic edge logs** (`gen_logs.py`): W3C/CLF-style fields (timestamp, client IP, method, path, status, bytes, UA, referrer, edge POP, WAF action) with injected, **ground-truth-labelled** SQLi, XSS, path traversal, credential-stuffing bursts and scanner sweeps over benign traffic. Banner everywhere: *"synthetic data"*.
- **Task A — triage:** 200-line window → few-shot prompt → bullet findings (top offending IPs/paths/attack classes). Auto-check against ground truth.
- **Task B — WAF rule drafting:** few-shot (attack line → regex) → generated regex. Validator: compiles with `re`; recall on labelled attack lines; false-positive rate on benign lines.
- **Why this showcases the drafter:** log lines and few-shot formats are highly repetitive (same paths, UAs, field layout) → suffix-match drafts should hit high α. Say so — and show XSUM as the *harder* case.
- **Privacy proof:** run with the network interface down; UI over a direct laptop↔Nano link.
- OPT-1.3B is a base model, so the prompt format is strict (`### LOGS … ### FINDINGS …`) and the validator filters bad output; the demo should show the validator, not hide it.

### 3.6 Benchmark matrix (final results table)

| Mode | Weights | KV | Spec | Workloads |
|---|---|---|---|---|
| A (baseline) | 4-bit | full | off | XSUM, CNN/DM, synthetic logs |
| B | 4-bit | H₂O | off | same |
| C | 4-bit | full | on | same |
| **D (EdgeSpec-H2O)** | 4-bit | H₂O | on | same |

Per mode: tok/s (prefill/decode), mean accepted tokens/step, α̂, peak RAM, KV bytes, ROUGE-2, PPL, WAF-rule recall/FPR, temps.

### 3.7 GATE 3 — must all pass on the Nano

| ID | Check | Pass criterion | Level |
|---|---|---|---|
| G3.1 | Sampler math | Toy test: 10⁵ samples from the speculative sampler vs. target p — chi-square p > 0.01 at l = 1; acceptance monotonically non-decreasing as l decreases | MUST |
| G3.2 | Spec correctness (greedy, **eviction off**, l = 1) | 20 prompts × 128 tokens: output identical to non-speculative greedy, allowing ≤ 1 % token mismatches; every mismatch must be a near-tie (top-2 logit margin < 0.05) confirmed with an FP32 re-check | MUST |
| G3.3 | Full system quality | Mode D vs Mode B at l = 1: ROUGE-2 and PPL equal within noise; Mode D vs Mode A within Gate-2 thresholds | MUST |
| G3.4 | Speed | Mode D vs Mode A decode tok/s: **≥ 1.5× (pass) · 2.0× (target) · 2.8× (stretch)** on at least the log workload; α̂, β, speedup reported for every workload, including the ones that miss | MUST (≥ 1.5× on logs) |
| G3.5 | Memory | Mode D peak RAM ≤ Phase-2 peak + 50 MiB; 30-min soak without OOM or swap growth | MUST |
| G3.6 | Lenience curve | α̂, speedup, and output divergence vs l = 1 for l ∈ {1.0, 0.8, 0.6, 0.45, 0.3} committed | MUST (report) |
| G3.7 | UI | Live stream ≥ 5 updates/s, slows decode ≤ 5 %; controls change cache/λ/l live; 3 consecutive dry runs without crash | MUST |
| G3.8 | Akamai demo (provisional, recalibrate after first run) | 20 synthetic windows: regex compile rate 100 %; attack recall ≥ 80 % and benign FPR ≤ 5 % on labelled lines; triage names the true top offender in ≥ 80 % of windows | SHOULD |
| G3.9 | Freeze | Tag `phase3-pass`; README with exact reproduction commands; results CSVs; deck numbers updated | MUST |

---

## 8. Fallback ladder (invoke after the 90-min fix timebox)

| Failure | Fallback |
|---|---|
| Fused CUDA kernel not working | Keep reference dequant path (already gate-passing) |
| 1.3B 4-bit < 1 tok/s or unstable | Switch to **OPT-350M** (GPTQ 4-bit PPL 24.24 vs 22.00 FP16 per Table 3), keep the same code |
| H₂O at 20 % fails quality at 1.3B | Carry the smallest passing budget (30–50 %); still a 2–3× KV cut; present the trade-off curve |
| λ < 1 doesn't beat λ = 1 | Ship λ = 1; present it as a tested-and-rejected extension |
| Verify cost β > 0.25 | Cap γ ≤ 2–3, reframe headline around H₂O + memory, report speculation honestly |
| α low on XSUM | Lead with logs/RAG/few-shot; show XSUM as the hard case |
| Nano OOM / throttling | Cap ctx at 1024, add fan, drop to 10 W→5 W only as a last resort and re-baseline |
| UI too heavy | Telemetry as JSON stream + static HTML on the laptop only |

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PyTorch CUDA context eats > 1.6 GB | Med | Med | Measure in G1.7; ctx 1024 fallback |
| Python overhead dominates step time on A57 | High | Med | Profile in G1.8; spec decoding amortizes it; consider CUDA graphs |
| FP16 nondeterminism breaks "lossless" tests | High | Low | Near-tie rule in G3.2; FP32 re-check |
| Tokenizer deps unavailable for Py3.6/aarch64 | Med | Low | Vendored BPE; laptop-side tokenization |
| Scope creep (Lexico, TinyLlama, UI polish) | High | High | Out of scope until Gate 3 passes |
| Demo model output too weak | Med | Med | Few-shot + validator; recorded fallback video |

## 10. Time map (24 h)

| Hours | Block |
|---|---|
| H0–H1 | Repo, env checks, Nano access, start offline GPTQ |
| H1–H8 | **Phase 1** build → **Gate 1** (H7–H8) |
| H8–H14 | **Phase 2** build (laptop sweep first) → **Gate 2** (H13–H14) |
| H14–H21 | **Phase 3** build → **Gate 3** (H20–H21) |
| H21–H24 | Freeze: benchmark matrix, slides with measured numbers, demo rehearsal ×3, recorded fallback |

Parallel prep that does **not** implement the next phase (your call whether to allow it): synthetic log generator, UI mock with fake telemetry, slide updates, eval scripts.

## 11. Open items from you

1. Confirm the Nano config above is current (or paste `nvpmodel -q`, `free -m`, `pip list | grep -i -E "torch|numpy"`).
2. How will you reach the Nano during the event — Wi-Fi/Ethernet SSH (recommended) or USB?
3. Is a laptop/Colab GPU available for offline GPTQ and the Phase-2 sweeps?
4. Fan and a 5 V 4 A barrel supply on the board?
5. OK to treat **OPT-1.3B as the only required model** (TinyLlama = stretch)?
