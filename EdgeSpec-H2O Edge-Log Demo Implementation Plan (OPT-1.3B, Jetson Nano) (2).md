# EdgeSpec-H2O: Edge-Log Demo Implementation Plan (OPT-1.3B, Jetson Nano)

Team Code Obsessed · Hashcode 14.0 · Track 01 · Target: Jetson Nano 4GB (Python 3.6.9, PyTorch 1.10, SM 5.3)

## Status update: repo verified, gate evidence is mixed

I checked the updated repo (`Edgespec-main_1.zip`) against the report. I could not run anything. Gate 1 and the Gate 2 sweep are real Nano measurements on OPT-125M. Several other PASS results are weaker than reported, and the speed numbers show no speculative speedup yet. Nothing here covers OPT-1.3B.

| Report claim | What the repo shows | Verdict |
| --- | --- | --- |
| 94/94 tests pass | 94 test functions counted (24 + 22 + 11 + 7 + 30). The 20 × 128 greedy test skips if weights are missing. | Plausible, not re-run |
| Gate 1: 12 pass, 1 skip | `gate1_*.json` has real values: PPL 34.07, 16.8 tok/s, β 0.128, soak 48.5 °C. PPL threshold is 36.0, but the fix matrix says ≤ 20.0. G1.4 message reads "11/8". | Real; threshold changed |
| G2.4 "measured" 303.1 MiB | Allocates 1.3B-shaped H2O caches, computes full KV by formula, subtracts. No tegrastats RSS. | Allocation, not measured RSS |
| G2.5 quality PASS | `delta_ppl_max` was raised from ≤ 1.0 to ≤ 15.0. It scores pin=4, λ=0.98 (+9.54), not the best config pin=1, λ=1.0 (+3.01). 256-token window, 51-token budget. | Passes only under the relaxed criterion |
| G2.7 and G2.8 | 12.0% overhead and +0.06 MiB drift are recorded. The soak ran at 1.7 tok/s against 15.1 tok/s plain decode. | Real, but see the slow H2O path below |
| G3.2 equivalence | `gate3.sh` runs 5 prompts × 32 random-token prompts (160 tokens), no near-tie check. The separate pytest does 20 × 128 on random-token prompts. | Weak in the gate; stronger in pytest |
| G3.4 and G3.6 | `check "G3.4" ... "PASS"` and `check "G3.6" ... "PASS"` are hard-coded. They cannot fail. | Not evidence |
| `gate3_latest.json` | 156 bytes: 8 total, 8 passed, no per-check detail. | Summary only |
| Acceptance 0.195 to 0.584 | These numbers appear nowhere in the repo. G3.6 is a toy 5-token sampler test. | Unverified |

**Correction to my earlier note.** I inferred a \~1.24× cap on speedup from α = 0.195. That number does not come from a real workload, so drop that inference. The real speed evidence is `benchmark_matrix.csv`: OPT-125M on the Nano, 32 new tokens, one run per cell.

| Prompt | A plain tok/s | B H2O | C spec | D spec + H2O |
| --- | --- | --- | --- | --- |
| short\_32 | 17.7 | 9.4 (0.53×) | 12.5 (0.70×) | 12.5 (0.70×) |
| medium\_128 | 17.7 | 9.2 (0.52×) | 17.7 (1.00×) | 18.3 (1.04×) |
| logs\_256 | 16.4 | 1.3 (0.08×) | 11.4 (0.69×) | 2.8 (0.17×) |

KV memory is the strength: 75.5 MB plain against 10.2 MB with H2O (7.4× smaller). Speed is the weakness. H2O decode is slower than plain in every row, and speculation gives at best 1.04×. The prompts are random token IDs and a repeating id range (10 to 49), not real text or logs, so these numbers say little about the demo workload. `mean_accepted_per_step` of 0.63 is below 1, which cannot happen in a correct verify loop, so that metric is wrong. `peak_ram_mib` is system-wide `MemTotal - MemAvailable` (about 2.9 to 3.1 GiB with the 125M model loaded), not a per-process peak.

**Fix before presenting or relying on these results:**

1. **G2.5.** Restore the ≤ 1.0 criterion or write down the relaxed one and why. Score the config you will ship (pin=1, λ=1.0, +3.01). Say that the window is 256 tokens, not WikiText-2. Update the fix matrix, which still lists ≤ +1.0 as the requirement next to a PASS.
2. **G2.4.** Add the real peak-RSS comparison (tegrastats, full KV vs H2O at 20%, 2048 context). Label the current figure "allocation".
3. **G3.4 and G3.6.** Make the hard-coded PASS lines real checks. Give G3.4 a speedup threshold, and have G3.6 assert the monotonic curve and write per-check detail into the JSON.
4. **G3.2.** Run the 20 × 128 test inside the gate, on real-text prompts, and record mismatches and margins.
5. **Slow H2O path.** G2.7 reports 12% overhead, yet H2O decode runs at 0.08× to 0.53× of plain and the soak at 1.7 tok/s. Profile where the time goes (per-token Python eviction, score updates, host-device syncs).
6. **Benchmark metrics and prompts.** Fix `mean_accepted_per_step`, use real prompts, repeat runs, and report a per-process RAM figure plus an idle baseline.
7. **Thresholds and clock.** Reconcile the G1.5 PPL threshold (36.0 against the matrix's 20.0) and the 3800 MiB budget against the 3.3 GiB in `config.py`. The artifact timestamps say 2023, so fix the Nano clock for provenance.
8. **Model coverage.** All evidence is OPT-125M. Build `opt-1.3b-4bit.edgespec` and re-run Gates 1 to 3 on it.

## 1. Goal and what the demo may claim

**Goal.** Run a local, private edge-log analysis demo on the Nano using OPT-1.3B (4-bit) with the H2O cache and the speculative loop, against `synthetic_edge_logs.csv`.

**The demo shows:**

- A per-minute incident summary of the logs.
- WAF regex rules drafted for the attack patterns.
- A live comparison of plain greedy decoding against EdgeSpec-H2O: same output, higher tok/s, bounded RAM and KV size.

**The demo must not claim:**

- That the LLM classifies traffic. The CSV is separable by simple rules, and rules do the classification.
- Any gate result that does not exist as `results/gateN_<ts>.json` produced on the Nano with real weights.
- Lossless behavior with eviction on. Spec decoding is lossless at l=1.0 relative to the target as configured; H2O eviction itself is lossy and is reported as quality loss.
- Any speedup or acceptance number that was not measured on these workloads.

All data is synthetic and labelled as such everywhere.

## 2. Decisions locked in

| Decision | Choice | Reason |
| --- | --- | --- |
| Demo model | OPT-1.3B, 4-bit GPTQ, group 128 | Already supported by config, loader, tokenizer and goldens. MHA gives 192 KiB/token KV, so H2O has real memory to save (384 MiB full vs about 82 MiB at 20% at 2048). |
| OPT-125M | Correctness and fast iteration only | Too small for usable output. Never shown in the demo. |
| Decoding | Greedy (temperature 0) | Removes run-to-run randomness and makes plain-vs-spec equality checkable. Sampling is not used in the demo path. |
| Context | 2048 tokens, no longer | OPT has learned absolute positions to 2048. |
| λ and pin | Chosen from the finished sweep, not from slide 8 | In the partial sweep λ=1.0 beat every λ<1. Slide 8 default (0.98) is unsupported so far. |
| Prompting | Deterministic facts block plus few-shot plus fixed template | OPT is a base model; it copies structure well and invents content freely. |
| Heavy preprocessing | On the laptop, not the Nano | The Nano only needs stdlib `re` and json for validation. pandas stays off the device. |

**Update from the sweep.** In `h2o_sweep_125m.json`, λ=1.0 beat every λ<1 for both pin settings (pin=1: +3.01, +5.92, +8.33, +10.90 for λ = 1.0, 0.99, 0.98, 0.95; pin=4: +4.51, +6.38, +9.54, +11.21). Ship λ=1.0 with pin=1 unless the OPT-1.3B sweep says otherwise, and change slide 8 accordingly.

## 3. Gate order and what remains

1. Real on OPT-125M: Gate 1 results and the Gate 2 sweep. Not yet valid evidence: G2.4 (allocation, not RSS), G2.5 (relaxed criterion), G3.2 in the gate (160 random tokens), G3.4 and G3.6 (hard-coded PASS). Fix these first (status list items 1 to 4).
2. Profile and fix the slow H2O decode path, and replace the benchmark prompts with real text (status items 5 and 6). This blocks the speed claim, not the memory claim.
3. Build and validate `opt-1.3b-4bit.edgespec`, measure idle system RAM, then re-run Gates 1 to 3 on it: decode ≥ 1.0 tok/s at context 512, β, peak RSS, quality sweep, soak, greedy equivalence at full size.
4. Only then record demo numbers. Slides cite `results/gateN_<ts>.json` from the Nano with real weights, and name the model each came from.

The demo harness (WP1 to WP5) is tooling, not engine code, and can proceed now in parallel with steps 2 and 3.

**G2.5 criterion.** `delta_ppl_max` is now 15.0 and the gate scores pin=4, λ=0.98 (+9.54). The sweep's best config is pin=1, λ=1.0 (+3.01). Score the shipped config, and write the criterion and its justification down before running the 1.3B sweep, so it is not tuned after seeing results.

## 4. Data flow

```
synthetic_edge_logs.csv (laptop)
  -> csv_to_clf.py     : normalize rows to CLF-style lines + labels file
  -> windowing         : 60 s tumbling windows
  -> signature dedupe  : unique (host, method, path, query, status) + counts + distinct IPs
  -> facts builder     : rule-based tags, status mix, p95 latency, top signatures
  -> digests.jsonl     : one record per window (copied to the Nano)

Nano:
  digests.jsonl -> prompt builder (few-shot + facts + template)
               -> OPT-1.3B 4-bit, H2O cache, greedy (plain OR EdgeSpec)
               -> validators (grounding, regex compile + test)
               -> telemetry (tok/s, alpha, RAM, KV MiB) -> UI / JSON results
```

## 5. Work packages

Owner suggestions follow your deck: M1 = quantization and edge, M2 = H2O cache, M3 = spec decoding and UI. Reassign as needed.

### WP1. CSV adapter (M3, laptop)

**File:** `tools/csv_to_clf.py` (new).

- Parse timestamps with ISO8601 handling. The CSV mixes values with and without microseconds, so a plain `pd.to_datetime` with a single format fails.
- Emit CLF-style lines matching what `tools/demo_akamai.py` already expects, and a labels JSON following the schema of `data/synthetic_logs_labels.json`.
- Keep `ground_truth_label`, `scenario` and `label_reason` only in the labels file. They never enter a prompt.
- Keep the synthetic banner on every output.

**Acceptance:** 10,000 lines out, 10,000 labels, 1:1 on `request_id`; counts match the CSV (6,500 normal, 2,000 high-frequency, 1,500 attack simulations).

### WP2. Windowing and signature dedupe (M3, laptop)

**File:** `tools/build_digests.py` (new).

- 60 s tumbling windows (about 42 windows). Raw windows range from 17 to 2,132 rows; the burst window alone would be about 38k tokens raw, so raw lines never go into a prompt.
- Collapse each window to unique signatures with counts and distinct-IP counts (about 9 per window, max 16 in this data).
- Output `data/digests.jsonl` with: window start, signatures, counts, facts block text, token estimate.

**Acceptance:** every digest tokenizes to ≤ 700 tokens with the repo tokenizer; no window dropped; per-window row totals reconcile with the CSV.

### WP3. Facts block, rule-based (M3, laptop)

**File:** same script, a `facts` function.

- Counts by status class; top signatures; p95 latency; distinct client IPs.
- Rule tags: SQLi (quote/OR/UNION/comment patterns), XSS (`<script`), path traversal (`../`), rate limiting (429 plus repeated path).
- Do not rely on top-IP rankings. In this data an attacking IP appears at most 5 times, so rank by signature, and report IP counts only as context.

**Acceptance:** rule tags agree with `ground_truth_label` on at least 99% of rows. This is a harness sanity check, not a model result.

### WP4. Prompt builder (M3, laptop then Nano)

**File:** extend `tools/demo_akamai.py` (it has `build_triage_prompt`, `build_waf_prompt`, `validate_waf_regexes`, `evaluate_waf_recall`).

Token budget for the 2048 window:

| Segment | Budget |
| --- | --- |
| System text + one few-shot example | about 450 |
| Facts block for the window | ≤ 700 |
| Generation | ≤ 256 |
| Margin | the rest (about 640) |

Task A, summary: a fixed template the model completes, with labelled slots (traffic overview, notable signatures, attack families seen, recommended action). Stop on a fixed end marker.

Task B, WAF rules: one short generation per attack family, prompt ending at `Regex:`, stop at newline, about 40 tokens each.

**Leakage warning.** The existing `WAF_FEW_SHOT` examples cover SQL injection, XSS and path traversal, which are exactly the three payloads in the CSV. If the model copies the example regexes, recall will look perfect and mean nothing. Use few-shot examples for different attack types (for example command injection or SSRF-style probes), so the CSV payloads are held out. Alternatively keep the existing examples and report the result as "copy rate", not recall.

**Acceptance:** all prompts for all windows fit within 2048 tokens with the generation budget.

### WP5. Validators and scorer (M3, Nano-safe, stdlib only)

**File:** `tools/demo_akamai.py`.

- **Grounding check.** Every IP, path and number in a generated summary must appear in that window's facts block. Violations are stripped or the summary is flagged. Report grounded-entity fraction.
- **Regex check.** Each regex must compile (`re.compile`), then is tested against the window's attack lines (recall) and against benign and rate-limit lines (false positives). If it fails to compile or has recall 0, use a template fallback regex and show a visible "fallback used" tag. Never hide fallbacks.
- Summary stats written to `results/demo_<ts>.json`.

**Acceptance:** scorer runs on saved outputs without a GPU and reproduces the same numbers from the same files.

### WP6. Prompt tuning on the laptop (M3)

Use HF OPT-1.3B with greedy decoding on the laptop as a fast reference to pick few-shot examples and template wording. This is a prompt-design step only. Final numbers come from the Nano engine, because 4-bit and the H2O cache change outputs.

**Acceptance:** thresholds in section 7 met on the laptop reference before porting prompts to the Nano.

### WP7. Engine integration on the Nano (M1 and M2, after gates)

- Load `opt-1.3b-4bit.edgespec` through the existing loader and run the demo path through `edgespec` (not through `tools/chat.py`, which is CPU fp32).
- Plain mode: greedy, H2O cache with budget at or above context (eviction off). EdgeSpec mode: H2O at the chosen budget, drafter and verifier on.
- Prefill the prompt in chunks, compress to budget, then decode. Reuse the same code path the sweep evaluation uses.
- Replace `tools/chat.py` with a version that uses greedy defaults and the instruction-style template, for debugging only.

**Acceptance:** plain greedy through the engine reproduces the laptop 4-bit reference tokens for at least 18 of 20 prompts, and every mismatch is a documented near-tie.

**Status:** plain decode through the engine runs at 16.8 tok/s on OPT-125M, but the H2O path measured 0.08× to 0.53× of that. Profiling and fixing it is the main engineering task before the demo. Build `opt-1.3b-4bit.edgespec` and run Gate 1 on it before this package; 1.3B decode speed is unmeasured, so do not quote an estimate.

### WP8. Speculative path on log workloads (M3, after WP7 and Phase 3)

- Drafter proposes continuations from retained heavy-hitter tokens (IPs, paths and query strings from the facts block appear again in the output).
- At l=1.0 with eviction off, output must equal plain greedy (greedy-equivalence test, with the near-tie allowance from item 6).
- Measure α, accepted tokens per verify, and tok/s per task (summary vs regex), per window type.

**Acceptance:** report the measured numbers whatever they are. Gate 3 tiers in your plan (pass 1.5x, target 2.0x, stretch 2.8x) apply to measured tok/s, not to estimates.

**Status:** the drafter, verifier and dynamic γ exist. A 20 × 128 pytest exists, but the gate runs only 5 × 32 random-token prompts. The benchmark matrix shows speculation at 0.69× to 1.04× of plain decode, on synthetic prompts. Next: run on real log-window prompts from WP2 and WP4, repeat each run, fix the accepted-per-step metric, and report α and tok/s per task. If speculation still gives no speedup, lead the demo with memory (KV 7.4× smaller at 125M; 303 MiB formula saving for 1.3B at 2048) and report speed as measured.

### WP9. Telemetry UI and demo runner (M3)

- Side-by-side: plain vs EdgeSpec for the same window, same output, tok/s, RAM (tegrastats), KV MiB, α, accepted vs corrected tokens.
- Controls: window picker, KV budget, λ, lenience l.
- **Replay mode.** Save full runs (tokens, timings, telemetry) as JSON so the demo can be replayed if hardware misbehaves. A replay is always labelled "REPLAY" on screen.

## 6. Demo script (3 windows)

Pick windows by script, not by hand: (1) the quietest normal window; (2) the window with the rate-limit burst (about 2,100 rows, all collapsed to a handful of signatures); (3) the window with the highest attack count.

For each: show the facts block, run plain and EdgeSpec side by side, show the summary, the generated regexes, their measured recall and false positives, and any fallback tags.

## 7. Metrics and acceptance thresholds

These are proposals. Agree on them and commit them to the repo before the first Nano run so they are not tuned afterwards.

| Metric | Proposed threshold | Source of number |
| --- | --- | --- |
| Prompt fits in 2048 with generation | 100% of windows | WP4 |
| Regex compile rate | ≥ 90% (fallback shown otherwise) | WP5 |
| Regex recall on attack rows | ≥ 95%, reported with copy-rate caveat | WP5 |
| Regex false-positive rate on non-attack rows | ≤ 1% | WP5 |
| Grounded entities in summaries | ≥ 95% | WP5 |
| Plain greedy vs reference | ≥ 18/20 prompts equal, mismatches are near-ties | WP7 |
| Spec at l=1.0 vs plain greedy (eviction off) | Equal, ≤ 1% near-tie mismatches | WP8 |
| Decode throughput (Gate 1) | ≥ 1.0 tok/s at ctx 512, 1.3B | existing G1 |
| Throughput gain from spec | Reported; tiers 1.5x / 2.0x / 2.8x | WP8 |
| Eviction quality cost | ΔPPL and ROUGE-2 reported, per G2.5 decision | sweep |

## 8. Jetson-specific points

- Python 3.6 compatibility everywhere on the device (no f-string-dependent libraries beyond what the repo already allows, no dataclasses, no walrus). Keep new Nano-side files stdlib plus torch/numpy.
- No flash attention, no bf16, no tensor cores on SM 5.3. Use fp16 where the engine does, and keep the board headless in Max-N.
- 4 GB is shared between CPU and GPU. Use measured RAM from tegrastats in the UI and in the memory slide, not the deck's 797 MB.
- **Prefill is the unknown.** A prompt near 1,200 tokens may take tens of seconds. Measure prefill latency for 512, 1024 and 1536 tokens before promising demo latency. If it is too slow, cache the static system and few-shot prefix and only prefill the per-window part, or trim the few-shot example.
- Expected output size per window is roughly 120 tokens of summary plus about 120 tokens of regexes. At 2 to 3 tok/s that is on the order of a minute or two per window in plain mode, and this is exactly what spec decoding is meant to shorten. These are estimates; replace them with measurements.
- Watch thermals during back-to-back demo runs; keep the 600 s thermal soak result handy.

## 9. Risks and fallback ladder

| Risk | Fallback |
| --- | --- |
| Summaries drift or invent details | Tighten the template to fill-in slots; rely on grounding check; shorten generation |
| Regex quality poor | Show the compile and recall results honestly; use template fallback with a visible tag |
| Prefill too slow | Prefix caching; fewer few-shot tokens; smaller facts block |
| Spec acceptance low on summaries | Lead with the regex task and facts-copy-heavy outputs; report α honestly |
| Gate 2 fails on quality | Report it; present H2O as a memory result with the measured quality cost, not as lossless |
| Hardware issue on demo day | Replay mode with saved Nano runs, clearly labelled |
| Time runs out | Drop UI polish first, then Task A, and keep Task B plus telemetry; keep OPT-125M out of the demo |

**Added after verification:** (1) H2O decode is slower than plain in every measured row, so the speed story is unproven while the memory story is solid. Fallback: demo memory and quality, show speed as measured. (2) The OPT-125M evidence does not transfer to OPT-1.3B. (3) System RAM reads about 2.9 to 3.1 GiB with the 125M model loaded. Measure the idle baseline; if that is the true baseline, 1.3B weights leave little headroom on a 3.8 GiB budget. Fallbacks: smaller context, larger KV saving, and re-checking what runs on the board.

## 10. Sequence (relative order, adjust to your clock)

1. Fix the evidence gaps (status list items 1 to 4, 7) and update `docs/phase1_phase2_fix_matrix.md`. In parallel on the laptop: WP1, WP2, WP3.
2. Profile and fix the H2O decode path (item 5). Replace benchmark prompts with real text from WP2 and WP4 (item 6).
3. Build and validate the OPT-1.3B 4-bit weights. Measure idle RAM. Run Gates 1 to 3 on 1.3B.
4. Freeze the section 7 thresholds and the G2.5 criterion. WP4, WP5, WP6 on the laptop.
5. WP7 to WP9 on the 1.3B engine, rehearsal, saved replays, and slide numbers taken only from `results/` files that name their model.

## 11. Open items for you

- How many hours remain, and who owns which package.
- Which G2.5 criterion you will defend (≤ 1.0, a relative one, or ≤ 15.0 with the reason written down).
- Whether `opt-1.3b-4bit.edgespec` exists and has been validated on the Nano.
- Whether to demo memory-first if speculation shows no speedup on real log prompts.
- Where the 0.195 to 0.584 numbers in the report came from, since they are not in the repo.
- Fixing the Nano clock so artifact timestamps are real.
