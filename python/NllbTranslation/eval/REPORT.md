# NLLB-200-3.3B on CTranslate2: Quality and Throughput Evaluation

**Question (as originally posed):** Does int8 quantization (`OpenNMT/nllb-200-3.3B-ct2-int8`,
CTranslate2) degrade translation quality vs the fp16 original (`facebook/nllb-200-3.3B`, PyTorch),
as deployed in the OpenMPF NllbTranslation component?

**Answer:** No — quantization is a no-op for quality, established across nine languages. But the
evaluation reframed the question twice. The throughput win belongs to the *engine*, not the
quantization; and on H100 int8 is actually *slower* than fp16, which inverts the deployment
recommendation.

**Data:** OPUS TED2020 → English. Nine source languages, 5,000 sentence pairs each (fixed random
sample, seed 42), gold sentence-aligned references from TMX.

| Pair | Source | NLLB code | Corpus (TU) |
|---|---|---|---|
| ar-en | Arabic (MSA) | arb_Arab | 398k |
| bn-en | Bangla | ben_Beng | 10k |
| de-en | German | deu_Latn | 289k |
| fa-en | Persian (Western) | pes_Arab | 297k |
| fr-en | French | fra_Latn | 400k |
| pt-en | Portuguese | por_Latn | 320k |
| ru-en | Russian | rus_Cyrl | 380k |
| uk-en | Ukrainian | ukr_Cyrl | 203k |
| zh-en | Mandarin (Simplified) | zho_Hans | 393k |

**Setup:** sacrebleu 2.6.0; COMET `Unbabel/wmt22-comet-da`. Two hardware platforms:

- **RTX 5070 Ti (16 GB)**, 2026-07-25 → 07-28 — the original run. Artifacts in `results/`.
- **H100**, 2026-07-29 — full re-run, all nine pairs, both axes, plus the decomposition on every
  pair. Artifacts committed in `pipeline-results/` (hypothesis and sample files excluded for size).

Three experiments:

- **Axis A — intrinsic** (9 pairs × 5,000 sentences): per-sentence, beam 4 on both, sentence-splitting
  neutralized. Isolates the deployed-model variable.
- **Axis B — as-deployed** (9 pairs, H100): whole file through each branch's own shipped pipeline
  (document-level).
- **Engine-vs-quantization decomposition** (9 pairs × 1,000 sentences, H100): splits Axis A's single
  contrast into its two confounded halves — inference *engine* and *numeric precision*.

---

## Bottom line

1. **Intrinsic quality: int8 is equivalent to fp16 in all nine languages.** Every per-sentence delta
   is ≤0.34 BLEU and ≤0.15 COMET on 0–100 scales, with signs pointing both ways.
2. **Quantization is a no-op for quality — now established per-language, not just once.** With the
   engine held fixed, Δ(int8 − fp16) is non-significant on **all nine pairs**: BLEU p = 0.33–0.96,
   COMET p = 0.21–0.98. Precision can be chosen purely on speed and memory grounds.
3. **The throughput win is the *engine*, and its size is hardware-dependent.** CTranslate2-fp16 over
   Transformers-fp16 is **~2.3× on H100** but **6.3× on the RTX 5070 Ti**. Quoting "~6×" without
   naming the GPU overstates it for server hardware.
4. **On H100, int8 is *slower* than fp16 — on all nine pairs** (0.80–0.98× CT2-fp16; ~8.5% slower on
   average, 20% slower on Chinese). Combined with (2), int8's only remaining advantage is footprint.
   **On GPU, prefer fp16.**
5. **As-deployed segmentation is broken in *both* branches, on different languages.** int8's
   character splitter is catastrophic on Chinese (−10.0 BLEU, length ratio 0.551), but `develop`
   under-generates on Bangla (0.692) and Persian (0.764), where int8 wins by ~7 BLEU. Neither
   pipeline is fit for dense scripts as shipped, and "port `develop`'s splitter" is no longer a
   complete fix.
6. **Axis A is hardware-independent.** The H100 run reproduced the RTX 5070 Ti Axis A scores to three
   decimals on all nine pairs — the quality conclusions do not depend on the GPU.

---

## Axis A — intrinsic quality (controlled, the fair comparison)

Per-sentence, beam 4 on both, splitter neutralized. Δ = int8 − fp16. Significance = paired
bootstrap, 1,000 resamples. 5,000 segments per pair. **Identical on both hardware platforms.**

| Pair | BLEU fp16 | BLEU int8 | ΔBLEU (sig) | ΔchrF (sig) | COMET fp16 | COMET int8 | ΔCOMET (sig) | Identical outputs |
|---|---|---|---|---|---|---|---|---|
| ar-en | 40.71 | 40.83 | +0.12 (n.s.) | +0.05 (n.s.) | 84.88 | 84.90 | +0.02 (n.s.) | 71% |
| bn-en | 33.24 | 33.15 | −0.10 (n.s.) | −0.08 (n.s.) | 85.75 | 85.74 | −0.01 (n.s.) | 66% |
| de-en | 39.04 | 38.70 | **−0.34** (p<0.001) | −0.15 (p=0.026) | 85.94 | 85.87 | −0.07 (p=0.016) | 75% |
| fa-en | 36.17 | 36.19 | +0.02 (n.s.) | −0.02 (n.s.) | 84.90 | 84.93 | +0.02 (n.s.) | 70% |
| fr-en | 42.61 | 42.32 | **−0.29** (p<0.001) | −0.18 (p=0.002) | 86.13 | 86.09 | −0.04 (n.s.) | 75% |
| pt-en | 46.04 | 45.87 | −0.18 (p=0.048) | −0.08 (n.s.) | 87.54 | 87.50 | −0.04 (n.s.) | 78% |
| ru-en | 30.47 | 30.42 | −0.05 (n.s.) | +0.00 (n.s.) | 82.15 | 82.11 | −0.04 (n.s.) | 68% |
| uk-en | 33.47 | 33.35 | −0.11 (n.s.) | −0.02 (n.s.) | 83.17 | 83.15 | −0.02 (n.s.) | 70% |
| zh-en | 24.38 | 25.11 | **+0.73** (p<0.001) | +0.19 (p=0.07) | 81.44 | 81.59 | **+0.15** (p<0.001) | 58% |

(TER, where the length guard allowed it: zh 70.14→66.51 **int8 better by 3.6**, bn 56.46→56.27 and
fa 52.75→52.68 int8 better, fr 43.89→44.10 and uk 54.00→54.09 fp16 marginally better.
pt/ar/de/ru skipped — a segment exceeded the TER length guard.)

**Reading it.** The largest delta anywhere is 0.34 BLEU (de-en) — well inside the range where two
systems are considered equivalent, and an order of magnitude below the between-language spread
(BLEU 24–46). The deltas *disagree in sign*: int8 wins zh clearly (+0.73 BLEU, +0.15 COMET, −3.6 TER
— all significant), loses de/fr/pt by ~0.2–0.3, and is statistically tied on the other five. With
5,000 paired segments the bootstrap resolves differences this small as "significant," but
statistical significance here is not practical significance: a 0.3-BLEU / 0.07-COMET shift is not
visible in output quality.

The identical-output rate (58–78%) tracks language difficulty — the harder the language, the more the
two decoders diverge — but where they diverge, quality is a wash.

**Cross-hardware reproducibility.** The H100 run reproduced every Axis A figure above to three
decimals (largest divergence: pt-en int8 45.865 vs 45.867). Same samples, same seed, different GPU
and driver stack. Beam-4 decoding on these systems is deterministic enough that quality results
transfer between platforms — worth knowing before re-running anything for quality reasons.

Note that Axis A still compares **two things at once**: Transformers-fp16 vs CTranslate2-int8.
Engine and precision are confounded. The decomposition below separates them.

## Engine vs quantization — decomposing the Axis A contrast

Axis A's "fp16 vs int8" bundles two independent changes: the *inference engine* (HuggingFace
Transformers → CTranslate2) and the *numeric precision* (fp16 → int8). These are orthogonal —
CTranslate2 runs fp16 perfectly well — so a third system, **CTranslate2-fp16**, splits the contrast
cleanly. Run on all nine pairs, 1,000 sentences each, H100, all three systems per-sentence at beam 4.
Compute types self-verified at load (`float16` and `int8_float16`).

Δengine = CT2-fp16 − HF-fp16 (same precision, different engine).
Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Pair | CT2-fp16 BLEU | Δengine BLEU (p) | Δquant BLEU (p) | Δquant COMET (p) |
|---|---|---|---|---|
| ar-en † | 40.20 | +1.91 (p<0.001) † | +0.17 (0.37) | −0.00 (0.98) |
| bn-en | 33.55 | +0.19 (0.14) | +0.07 (0.71) | −0.04 (0.47) |
| de-en | 38.85 | −0.19 (0.36) | +0.00 (0.96) | +0.03 (0.71) |
| fa-en | 36.11 | +0.11 (0.17) | −0.05 (0.73) | +0.07 (0.21) |
| fr-en | 42.53 | −0.26 (0.036) | −0.18 (0.33) | −0.02 (0.59) |
| pt-en | 44.92 | −0.14 (0.46) | −0.03 (0.85) | −0.01 (0.84) |
| ru-en | 30.48 | +0.15 (0.076) | −0.14 (0.34) | −0.04 (0.48) |
| uk-en | 33.46 | −0.10 (0.29) | +0.06 (0.67) | +0.03 (0.61) |
| zh-en | 25.55 | +0.08 (0.58) | −0.08 (0.58) | −0.01 (0.86) |

† **ar-en is an unexplained outlier.** It is the only pair where Δengine is large (+1.91 BLEU /
+0.71 COMET, p<0.001); the other eight span −0.26 to +0.19 and reach significance only for fr-en.
It also has the slowest HF-fp16 throughput by a wide margin (1.23 vs ~2.2 sent/s), giving it the
highest apparent engine speedup (4.1× vs ~2.3×).

An earlier version of this report attributed both to a harness confound —
`run_decomp.sh` passed no job properties to the HF system, leaving
`DIFFICULT_LANGUAGE_TOKEN_LIMIT=50` active for Arabic, while `run_pipeline.sh` disables it for
Axis A. **That explanation was tested and is wrong.** Re-running the ar-en decomposition with
`DIFFICULT_LANGUAGE_TOKEN_LIMIT=0` produced **identical** results. (`run_decomp.sh` passes the
property for parity now regardless; the change turned out to be harmless rather than necessary.)

A second explanation — that the sentences are too short to reach the 50-token limit — also fails:
**13.4% of ar-en sentences exceed 50 tokens** (median 25, p90 56, max 541). Why the limit is inert
is not established. One untested possibility is that the splitter cannot subdivide a *single*
sentence, making any token limit a no-op for the decomposition's per-sentence input.

The figure is therefore reported as measured and **included** in the engine range, flagged as
unexplained rather than dismissed. If the engine effect is genuinely language-dependent, that is
worth knowing; a repeat on a second Arabic-script corpus would settle it.

**Quantization is a no-op for quality.** Δquant is non-significant on **every pair, on both metrics**
(BLEU p = 0.33–0.96, COMET p = 0.21–0.98), with signs mixed — even though the two systems genuinely
produce different text (214/1,000 outputs differ on pt-en). This upgrades the earlier single-language
result into a nine-language one, and it is the finding that makes precision a pure
engineering choice.

### Throughput (sentences/sec, single-sentence latency, batch 1, H100)

| Pair | HF-fp16 | CT2-fp16 | CT2-int8 | engine speedup | int8 ÷ fp16 |
|---|---|---|---|---|---|
| ar-en | 1.23 † | 5.02 | 4.65 | **4.1×** † | 0.93× |
| bn-en | 2.15 | 5.06 | 4.66 | 2.4× | 0.92× |
| de-en | 2.23 | 4.80 | 4.44 | 2.2× | 0.93× |
| fa-en | 2.05 | 4.62 | 4.55 | 2.3× | 0.98× |
| fr-en | 2.06 | 4.95 | 4.75 | 2.4× | 0.96× |
| pt-en | 2.06 | 5.00 | 4.59 | 2.4× | 0.92× |
| ru-en | 2.23 | 5.33 | 4.77 | 2.4× | 0.89× |
| uk-en | 2.22 | 5.30 | 4.84 | 2.4× | 0.91× |
| zh-en | 2.73 | 4.64 | 3.69 | 1.7× | **0.80×** |

**Two conclusions, both of which change prior guidance.**

**The speedup is the engine, and it is smaller on server hardware.** CTranslate2 at *unchanged* fp16
precision buys 1.7–2.4× on H100 across eight of nine pairs, and 4.1× on the ar-en outlier (†). The same contrast measured 6.3× on the RTX 5070 Ti: H100
accelerates the batched PyTorch path far more than it accelerates CTranslate2's latency-bound
single-sentence path. The engine win is real and worth taking, but "~6×" is a consumer-GPU number.

**int8 is slower than fp16 on H100, unanimously.** Every pair lands below 1.0×, averaging ~0.92× and
falling to 0.80× on Chinese. This reverses the RTX 5070 Ti result (int8 was ~5% *faster* there).
Since quantization is also quality-neutral, int8 retains exactly one advantage on GPU — 3.36 GB vs
6.7 GB — which is immaterial on an 80 GB card. **CT2-fp16 is the right GPU configuration.** int8
remains correct for CPU deployment, where CTranslate2 does not support fp16 at all.

### A note on the main pipeline's own timings

The pipeline compares **batched fp16** against **per-line int8**, so its throughput numbers are not
like-for-like — and the asymmetry flipped direction between platforms:

| Platform | int8 (per-line) | fp16 (batched) | apparent winner |
|---|---|---|---|
| RTX 5070 Ti | 3.32–3.66 | 0.74–1.11 | int8, by 3.3–4.7× |
| H100 | 2.75–3.62 | 5.89–7.98 (batch 16–24) | fp16, by ~2.2× |

int8's per-line rate barely moved between platforms (~3.3 vs ~3.1 sent/s) because batch-1 decoding is
latency-bound, while the batched fp16 path scaled with the GPU. Read these as a batching story, not
a precision story; the batch-1 table above is the controlled comparison. (pt-en's H100 metadata
predates the throughput fields and is excluded.)

## Axis B — as-deployed (each branch exactly as shipped)

Whole 5,000-sentence file as one document through each branch's own pipeline (fp16: **greedy** +
`sat-3l-sm` token-based splitter; int8: **beam 4** + `wtp-bert-mini` **character-based** splitter).
All nine pairs, H100.

| Pair | BLEU fp16 | BLEU int8 | ΔBLEU | ratio fp16 | ratio int8 |
|---|---|---|---|---|---|
| ar-en | 46.19 | 45.18 | −1.00 | 0.942 | 0.909 |
| **bn-en** | 29.86 | **37.43** | **+7.57** | **0.692** | 0.825 |
| de-en | 45.97 | 45.24 | −0.73 | 0.915 | 0.906 |
| **fa-en** | 34.98 | **42.20** | **+7.22** | **0.764** | 0.899 |
| fr-en | 49.35 | 48.42 | −0.93 | 0.931 | 0.911 |
| pt-en | 51.47 | 49.96 | −1.51 | 0.935 | 0.908 |
| ru-en | 38.74 | 38.54 | −0.20 | 0.915 | 0.904 |
| uk-en | 41.78 | 41.16 | −0.62 | 0.915 | 0.905 |
| **zh-en** | 27.03 | **17.00** | **−10.03** | 0.751 | **0.551** |

**This is not the story the three-language run told.** With all nine pairs measured, the pattern is
**bimodal**, and the earlier conclusion — "int8's character splitter is the whole gap" — is only
correct for Chinese.

- **Chinese: int8's splitter is the problem, confirmed and worse.** −10.03 BLEU at a length ratio of
  0.551. The int8 branch splits on *character count* (360 chars); Chinese is spaceless and
  character-dense, so 360 characters is an enormous span → few very long chunks → the model
  under-translates them.
- **Bangla and Persian: `develop` is the problem.** Look at the fp16 column — 0.692 and 0.764. The
  supposedly-better token-based pipeline under-generates badly on these scripts, and int8 wins by
  ~7 BLEU. `develop`'s splitter is not uniformly superior.
- **Latin/Cyrillic: a consistent small int8 deficit** (−0.20 to −1.51), tracking slightly lower
  ratios (0.904–0.911 vs 0.915–0.942). The character splitter costs a little everywhere.

**Axis B confounds two variables and cannot attribute this on its own.** The two branches differ in
*both* segmentation *and* decoding — `develop` decodes **greedily** (its `generation_config.json`
sets no `num_beams`), while the int8 branch uses **beam 4**. Axis A rules out the model itself
(bn −0.10, fa +0.02 with decoding harmonized and splitting neutralized), so the bn/fa reversal comes
from the pipeline — but whether it is greedy decoding truncating output or the splitter mishandling
those scripts is not separable from these artifacts. Greedy decoding is the leading hypothesis, and
it is cheap to test: re-run `develop`'s blob with beam 4 forced (the eval driver already does this at
runtime via `generation_config.num_beams = 4`, no rebuild needed). If the gap closes, it was
decoding.

---

## Follow-up: the as-deployed gap is chunk size, and it is fixable

*(Added 2026-07-31, after the CTranslate2 implementation work. Supersedes the splitter
recommendation below, which is left in place as the reasoning at the time.)*

Porting `develop`'s token-based splitter to the CTranslate2 branch fixed Chinese as-deployed
(length ratio 0.551 -> 0.704) but **regressed** Bangla (0.825 -> 0.666) and Persian (0.899 -> 0.756):
it imported `develop`'s own under-generation. Neither splitter was right.

A controlled sweep (`eval/sweep_splitter.sh`, 200 sentences/pair) found the actual mechanism.
**NLLB under-generates on long inputs, monotonically in chunk count** - on bn-en: 15 chunks ->
0.363, 23 -> 0.457, 42 -> 0.727, 69 -> 0.869, 238 -> 1.031. Raising the token budget makes it worse.

Setting **`SENTENCE_SPLITTER_MODE=SENTENCE`** - one sentence per chunk - fixes every pair tested:

| pair | packed chunks | **one sentence per chunk** |
|---|---|---|
| bn-en | 23.73 / 0.727 | **33.60 / 1.031** |
| zh-en | 16.84 / 0.704 | **28.24 / 0.986** |
| pt-en (Latin control) | 48.25 / 0.949 | **48.85 / 0.971** |

Every ratio lands in 0.97-1.03 - better than `develop`'s best (0.94) and better than the character
splitter's best (0.899), with no regression on the Latin control. It is now the shipped default.

Two consequences for how this report should be read:

- **The character-vs-token splitter framing was a red herring.** With one sentence per chunk the
  sizing unit barely matters (`USE_NLLB_TOKEN_LENGTH=FALSE` scores the same). The -10.0 BLEU Chinese
  result and the +7 BLEU Bangla/Persian results were both artifacts of *packing sentences into
  chunks*, not of characters versus tokens.
- **This explains why Axis A never showed under-generation.** Axis A feeds one sentence per
  detection - it was unknowingly running the fixed configuration all along, which is why the
  intrinsic and as-deployed axes disagreed so sharply.

---

## Recommendation

1. **Choose precision by device, not by quality.** Quantization is quality-neutral on all nine
   languages, so the decision is purely device fit:
   - **GPU → fp16.** Faster than int8 on every pair on H100, and the 2× footprint is immaterial on
     server cards.
   - **CPU → int8** (`int8_float32`). CTranslate2 does not support fp16 on CPU at all; a
     float16 model loaded on CPU is silently up-converted to float32, forfeiting the size win.
2. **Adopt CTranslate2 as the engine** — ~2.3× over Transformers on H100 at indistinguishable
   quality, and more on smaller cards. Credit the gain to the engine, not to quantization.
3. **Fix segmentation for dense scripts — in both directions.** Porting `develop`'s token-based
   splitter remains necessary for Chinese, but it is **not sufficient and not risk-free**: it would
   import `develop`'s bn/fa under-generation. Gate any splitter change on Axis B for **zh, bn, and
   fa**, and target a length ratio near 0.90 rather than `develop`'s 0.751.
4. **Keep beam 4; never inherit `develop`'s greedy default.** It is the most likely cause of the
   ~7 BLEU bn/fa deficit and costs little on this engine.
5. ~~**Fix `NLLB_MODEL` handling on the ctranslate2 branch (component bug, found during this
   work).**~~ **RESOLVED** on `prototype/nllb-ctranslate2` (commit `777437f9`). The component loaded
   `DEFAULT_NLLB_MODEL` in `__init__` and `_check_model` only reloaded
   `if not self._model.model_is_loaded` — never false for a live `Translator` — so a job-level
   `NLLB_MODEL` was silently **ignored** and the baked-in model served every request. This is the
   defect that invalidated a decomposition run and forced the standalone `ct2_driver.py` workaround.
   `_check_model` now compares the requested name against the loaded one and reloads, resetting the
   tokenizer to the new model directory; a missing directory raises rather than falling back.
   Verified by switching models between jobs and by a bogus name.
   *Note:* `run_decomp.sh` still uses `ct2_driver.py`, which remains useful for forcing a
   `compute_type` and recording `actual_compute_type` — but the bug that made the bypass
   *necessary* is gone.
6. **Assert the resolved `compute_type` at load.** Precision is becoming build-time-conditional, and
   two silent failure modes exist: bare `int8` resolving to the slow `int8_float32` on GPU, and
   float16 up-converting to float32 on CPU. One log line makes the deployed numerics auditable.

Implementation plan: `../PLAN.md`.

## Caveats

- **Axis B confounds splitter and decoding** (greedy vs beam 4). The bn/fa reversal is established;
  its cause is not. See the discriminating experiment above.
- **The ar-en engine contrast is unexplained**, not invalid. The chunking-confound explanation was
  tested and disproved (re-running with `DIFFICULT_LANGUAGE_TOKEN_LIMIT=0` gave identical results),
  and "sentences too short to trigger it" also fails (13.4% exceed 50 tokens). Reported as measured
  and included in the range; see the † note under the decomposition table.
- **Decomposition sample size** is 1,000 per pair (vs 5,000 for Axis A), so its confidence intervals
  are wider. The quantization null result is consistent across all nine pairs, which is what carries
  it, rather than any single pair's precision.
- **Bangla corpus is small:** the bn-en TMX holds only 10,260 units, so the 5,000-sentence sample is
  ~49% of the available corpus. The paired comparison is unaffected, but the bn sample is less
  independent of the corpus than the others.
- **Per-language difficulty:** absolute scores vary by language (zh hardest at BLEU ~25; that's the
  language, not the engine). Arabic is NLLB's flagged "difficult" language; its special split limit
  was disabled in Axis A so both branches segment it identically.
- TED2020 references are loose crowd translations (caps absolute BLEU); possible pretraining
  contamination inflates absolute scores. Both affect fp16 and int8 equally — the deltas stand.
- **ru-en's fp16 leg was resumed** on the RTX 5070 Ti run after a CUDA error at 3,992/5,000.
  Hypotheses and scoring are complete over all 5,000 segments; its throughput figure covers only the
  resumed portion and is excluded from timing ranges.
- **fa-en needed a language-code fix first.** The initial run was skipped by the fail-fast smoke test
  because the configured code was `per_Arab` — `per` is ISO 639-2/B, and FLORES-200 uses ISO 639-3.
  Verified against the NLLB tokenizer: `per_Arab` → UNK (id 3), **`pes_Arab` → id 256053**.
- **pt-en Axis A was re-sampled** from TMX (BLEU 46.04/45.87, Δ−0.18), replacing an earlier
  Moses-derived sample (46.47/46.23, Δ−0.24). Both agree: negligible, int8 marginally behind.

## Not yet run

- **CPU-target measurement.** int8 is recommended for CPU on capability grounds (fp16 is unsupported
  there), but NLLB-3.3B CPU throughput has not been measured. Whether the CPU target is viable for
  any real workload is an open question.
- **The beam-4 discriminating run** for bn/fa described above — the single highest-value remaining
  experiment, since it decides whether the splitter port is a fix or a regression.
- **Batched throughput for the CT2 systems.** All decomposition figures are batch 1. Batched
  CTranslate2 was spot-checked only on the RTX 5070 Ti (fp16 33.1 vs int8 32.1 sent/s).

## Artifacts

- **`pipeline-results/`** — H100 run (current). `SUMMARY.md`; per pair `axisA.report.txt`,
  `axisB.{fp16,int8}.report.txt`, `meta.*.json`; per pair `decomp/<pair>/decomp.SUMMARY.md` and
  `decomp.{engine,quant}.report.txt`. Hypothesis and sample files omitted for size.
- **`results/`** — RTX 5070 Ti run (original, gitignored locally). Adds `axisA.segments.csv`
  (per-sentence + COMET) and the `hyp.*.en` files.
- **Pipeline:** `run_pipeline.sh`, `run_decomp.sh`, `tmx_sample.py`, `mt_eval.py`,
  `nllb_eval_driver.py`, `ct2_driver.py`, `convert_ct2.sh`, `summarize.py`, `decomp_report.py`.
- **Timing** (RTX 5070 Ti, 5,000 sentences/pair): int8 23–25 min, fp16 1.3–2.1 h.
