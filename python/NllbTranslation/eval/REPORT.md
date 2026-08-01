# NLLB-200-3.3B on CTranslate2: Quality and Throughput Evaluation

**Question (as originally posed):** Does int8 quantization (`OpenNMT/nllb-200-3.3B-ct2-int8`,
CTranslate2) degrade translation quality vs the fp16 original (`facebook/nllb-200-3.3B`, PyTorch),
as deployed in the OpenMPF NllbTranslation component?

**Answer:** No — quantization is a no-op for quality, established across nine languages. But the
question turned out to be the wrong one, three times over. The throughput win belongs to the
*engine*, not the quantization; on H100 int8 is actually *slower* than fp16, which inverts the
deployment recommendation; and the only variable that moved as-deployed quality by more than a
rounding error was **neither** engine nor precision but **sentence segmentation**, worth up to
14 BLEU — two orders of magnitude more than the effect the study set out to measure.

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

**Setup:** sacrebleu 2.6.0; COMET `Unbabel/wmt22-comet-da`. Three runs across two hardware platforms:

- **RTX 5070 Ti (16 GB)**, 2026-07-25 → 07-28 — the original run. Artifacts in `results/`.
- **H100**, 2026-07-29 — full re-run, all nine pairs, both axes, plus the decomposition on every
  pair. Superseded; recoverable from git history at `ce0d5733`.
- **H100**, 2026-07-31 → 08-01 — the **final** nine-pair run, after the segmentation fix and the
  switch to a build-time fp16 conversion, plus a clean re-run of the decomposition on every pair.
  This is the configuration that ships and the source of every current figure; artifacts are
  committed in `pipeline-results/` (hypothesis and sample files excluded for size). See
  [the final results](#final-results--the-shipping-configuration-nine-pairs) below.

Four experiments:

- **Axis A — intrinsic** (9 pairs × 5,000 sentences): per-sentence, beam 4 on both, sentence-splitting
  neutralized. Isolates the deployed-model variable.
- **Axis B — as-deployed** (9 pairs, H100): whole file through each branch's own shipped pipeline
  (document-level).
- **Engine-vs-quantization decomposition** (9 pairs × 1,000 sentences, H100): splits Axis A's single
  contrast into its two confounded halves — inference *engine* and *numeric precision*.
- **Splitter sweep** (`sweep_splitter.sh`, 200 sentences/pair) and its **full-scale confirmation**
  (9 pairs × 5,000, H100): isolates chunk size, which turned out to dominate everything above.

---

## Bottom line

1. **Intrinsic quality: int8 is equivalent to fp16 in all nine languages.** Every per-sentence delta
   is ≤0.34 BLEU and ≤0.15 COMET on 0–100 scales, with signs pointing both ways.
2. **Quantization is a no-op for quality — now established per-language, not just once.** With the
   engine held fixed, Δ(int8 − fp16) is non-significant on **all nine pairs**: BLEU p = 0.33–0.96,
   COMET p = 0.21–0.98. Precision can be chosen purely on speed and memory grounds.
3. **The throughput win is the *engine*, and its size is hardware-dependent.** CTranslate2-fp16 over
   Transformers-fp16 is **~2.4× on H100** — 2.35–2.47× across nine languages, a 5% spread — but
   **6.3× on the RTX 5070 Ti**. Quoting "~6×" without naming the GPU overstates it for server
   hardware.
4. **On H100, int8 is *slower* than fp16 — on all nine pairs** (0.84–0.91× CT2-fp16, ~12% slower on
   average, no pair reaching parity). Combined with (2), int8's only remaining advantage is
   footprint. **On GPU, prefer fp16.**
5. ~~**As-deployed segmentation is broken in *both* branches, on different languages.**~~ **Superseded
   — and it was the finding that mattered.** As originally measured, the character splitter was
   catastrophic on Chinese (−10.0 BLEU, length ratio 0.551) while `develop` under-generated on Bangla
   (0.692) and Persian (0.764). The cause of *both* halves turned out to be the same thing —
   **packing multiple sentences into a chunk** — and one setting fixes it on every pair.
   See [Final results](#final-results--the-shipping-configuration-nine-pairs).
6. **The shipping configuration beats `develop` as-deployed on all nine pairs**, by **+1.23 to
   +14.03 BLEU**, having *lost* on seven of nine before the fix. Length ratios go from 0.55–0.91
   (CTranslate2, character splitter) and 0.69–0.94 (`develop`) to **0.946–1.024**.
   **That gain is length recovery, not better decoding** — a BLEU brevity-penalty decomposition puts
   the n-gram precision term at −1.48…+0.75 BLEU, negative on five of nine pairs, and ΔBLEU tracks
   Δlength-ratio at r² = 0.971. One mechanism, nine magnitudes.
7. **Axis A is hardware-independent.** The H100 run reproduced the RTX 5070 Ti Axis A scores to three
   decimals on all nine pairs — the quality conclusions do not depend on the GPU.
8. **The engine swap is quality-neutral at full scale.** The final nine-pair Axis A (fp16 vs fp16,
   engine as the *only* variable, 5,000 segments/pair) puts ΔBLEU in **−0.25 … +0.71**. The engine
   buys throughput and nothing else — which is exactly what you want from an engine swap.

---

## Axis A — intrinsic quality (controlled, the fair comparison)

> **Which Axis A is "the" Axis A?** This one contrasts **Transformers-fp16 vs CTranslate2-int8**, so
> engine and precision are confounded; it is what the study originally ran, and its conclusion
> (quantization is quality-neutral) still stands. The **shipping** configuration is fp16 on both
> sides, making the engine the sole variable — that table is in
> [Final results](#final-results--the-shipping-configuration-nine-pairs). The two agree, and both
> agree with the clean decomposition below.

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

**These are the 2026-08-01 clean re-run figures**, with `DIFFICULT_LANGUAGE_TOKEN_LIMIT=0` passed to
the HF leg on every pair and all `hyp.*.en` deleted first. Two pairs changed materially from the
first attempt; both were harness defects, documented under the table.

| Pair | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine BLEU | Δquant BLEU |
|---|---|---|---|---|---|
| ar-en | 40.236 | 40.198 | 40.370 | −0.038 | +0.172 |
| bn-en | 33.356 | 33.547 | 33.619 | +0.191 | +0.072 |
| de-en | 39.038 | 38.850 | 38.854 | −0.188 | +0.004 |
| fa-en | 36.002 | 36.109 | 36.058 | +0.107 | −0.051 |
| fr-en | 42.793 | 42.529 | 42.348 | −0.264 | −0.181 |
| pt-en | 45.062 | 44.923 | 44.896 | −0.139 | −0.027 |
| ru-en | 30.325 | 30.475 | 30.338 | +0.150 | −0.138 |
| uk-en | 33.562 | 33.459 | 33.522 | −0.102 | +0.063 |
| zh-en | 24.946 | 25.549 | 25.465 | **+0.603** | −0.085 |

**Δquant is non-significant on all nine pairs** (BLEU p = 0.33–0.96, COMET p = 0.21–0.98), spanning
−0.181…+0.172. Quantization is quality-neutral; precision is a speed-and-memory decision.

**Δengine spans −0.264…+0.603**, and only zh-en moves. That figure now *agrees* with the
independent n=5,000 Axis A measurement of the same contrast (+0.709, see
[Final results](#final-results--the-shipping-configuration-nine-pairs)); at n=1,000 it is not
individually significant (p=0.158, 95% CI [−0.13, +1.46]), but the two runs bracket the same effect.

### Two harness defects this run corrected

Both were found by comparing the re-run against the original artifacts, and both share the shape
described in the caveats: **absence of change reads as confirmation.**

**1. ar-en — the `DIFFICULT_LANGUAGE_TOKEN_LIMIT` confound (previously documented).** The original
run produced HF-fp16 38.288 and Δengine **+1.91**, the largest engine effect in the study. It was an
artifact: `run_decomp.sh` passed no job properties to the HF system, leaving the limit at its default
50, while `run_pipeline.sh` disables it. Arabic is the only language in `PROCESS_DIFFICULT_LANGUAGES`
by default, and 13.4% of the ar-en sample exceeds 50 tokens, so only that leg got sub-chunked.

| run | HF-fp16 BLEU | Δengine BLEU |
|---|---|---|
| original (limit active) | 38.288 | **+1.91** |
| clean, RTX 5070 Ti, n=200 | 40.81 | +0.155 |
| clean, H100, n=1,000 | 40.236 | −0.038 |
| **this run** | **40.236** | **−0.038** |

Throughput normalised with it (1.226 → 2.089 sent/s), so the apparent 4.1× engine speedup on this
pair was the same artifact.

**2. zh-en — the HF leg was scoring another system's output.** In the original run zh-en's HF-fp16
column was **identical to its CT2-int8 column on every metric** — BLEU 25.465, chrF 49.785,
chrF++ 48.019, TER 67.122, COMET 81.629, all to three decimals. Two different systems do not agree
to three decimals on five metrics; that is one hypothesis file scored twice. The algebraic
signature was visible in the published table as Δengine = −Δquant = 0.085 exactly.

The clean re-run gives HF-fp16 **24.946**, and Δengine moves +0.085 → **+0.603**. This also retires
an inference in an earlier revision of this report, which treated the n=1,000 Δengine column as
merely *underpowered* on zh-en and preferred the n=5,000 Axis A figure. The column was not noisy
there — it was wrong, for a specific and findable reason, and the corrected value agrees with
Axis A.

Two candidate mechanisms, both documented traps in this harness, and the artifacts do not
distinguish them: the crossed image-tag assignment that once existed in `run_decomp.sh`, or its
resume guard preserving a stale `hyp.hf-fp16.en` across every subsequent run. Either way the rule is
the same — **delete `hyp.*.en` before re-running anything.**

*Seven of nine pairs reproduced their BLEU exactly* (all metrics, three decimals) while their
throughput figures moved, confirming both that they were genuinely regenerated and that beam-4
decoding on this stack is deterministic.

### Throughput (sentences/sec, single-sentence latency, batch 1, H100)

Same clean re-run. Both defects above also distorted this table; corrected, it is strikingly uniform.

| Pair | HF-fp16 | CT2-fp16 | CT2-int8 | engine speedup | int8 ÷ fp16 |
|---|---|---|---|---|---|
| ar-en | 2.089 | 4.977 | 4.253 | 2.38× | 0.85× |
| bn-en | 2.206 | 5.179 | 4.574 | 2.35× | 0.88× |
| de-en | 2.248 | 5.318 | 4.660 | 2.37× | 0.88× |
| fa-en | 2.128 | 5.042 | 4.569 | 2.37× | 0.91× |
| fr-en | 2.056 | 5.081 | 4.440 | 2.47× | 0.87× |
| pt-en | 2.097 | 5.005 | 4.539 | 2.39× | 0.91× |
| ru-en | 2.252 | 5.386 | 4.498 | 2.39× | 0.84× |
| uk-en | 2.173 | 5.137 | 4.624 | 2.36× | 0.90× |
| zh-en | 2.105 | 5.079 | 4.576 | 2.41× | 0.90× |

**Two conclusions, both of which change prior guidance.**

**The speedup is the engine, and on H100 it is ~2.4× — uniformly.** The range is **2.35–2.47×**
across nine languages, a spread of 5%. The previously reported "1.7–2.4×, varies by language" was
itself an artifact of the two defects: ar-en's 4.1× came from the depressed HF baseline, and zh-en's
1.7× from the mis-scored leg. There is no real per-language variation in the engine win to explain.

It *is* hardware-dependent: the same contrast measured 6.3× on the RTX 5070 Ti, because H100
accelerates the batched PyTorch path far more than CTranslate2's latency-bound single-sentence path.
The engine win is real and worth taking, but "~6×" is a consumer-GPU number.

**int8 is slower than fp16 on H100, unanimously** — 0.84–0.91×, averaging ~0.88×, with no pair
reaching parity. This reverses the RTX 5070 Ti result (int8 was ~5% *faster* there). Since
quantization is also quality-neutral, int8 retains exactly one advantage on GPU — 3.36 GB vs 6.7 GB
— which is immaterial on an 80 GB card. **CT2-fp16 is the right GPU configuration.** int8 remains
correct for CPU deployment, where CTranslate2 does not support fp16 at all.

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

## Axis B — as-deployed (each branch exactly as shipped) — *superseded*

> **These figures no longer describe either branch.** They measure the CTranslate2 branch's *former*
> character splitter; the current one sets `SENTENCE_SPLITTER_MODE=SENTENCE` and every delta below
> changes sign. Retained because it is the evidence that started the segmentation investigation, and
> because its `develop` column is the unchanged baseline the final table is measured against.
> Current numbers: [Final results](#final-results--the-shipping-configuration-nine-pairs).

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

## Final results — the shipping configuration, nine pairs

*(2026-07-31 H100 run. This section is the authoritative result; everything above it is the path
that led here, retained as the reasoning at the time.)*

**CT2** = the `prototype/nllb-ctranslate2` image as it ships: `facebook/nllb-200-3.3B` converted at
build time, **float16** on GPU, beam 4, `SENTENCE_SPLITTER_MODE=SENTENCE`.
**HF** = the `develop` image: same checkpoint, Transformers fp16, greedy, packed token splitter.

Note what this changes about the contrast: both sides are now **fp16**, so Axis A isolates the
*engine* alone — no quantization confound, at 5,000 segments/pair rather than the decomposition's
1,000.

### Axis A — intrinsic, engine as the only variable

| Pair | BLEU HF | BLEU CT2 | ΔBLEU | ΔchrF | COMET HF | COMET CT2 | ΔCOMET |
|---|---|---|---|---|---|---|---|
| ar-en | 40.709 | 40.458 | −0.251 | −0.116 | 84.880 | 84.845 | −0.034 |
| bn-en | 33.243 | 33.269 | +0.026 | +0.062 | 85.755 | 85.832 | +0.077 |
| de-en | 39.037 | 38.879 | −0.158 | −0.057 | 85.939 | 85.928 | −0.011 |
| fa-en | 36.170 | 36.125 | −0.045 | −0.008 | 84.903 | 84.942 | +0.039 |
| fr-en | 42.614 | 42.586 | −0.028 | +0.006 | 86.131 | 86.137 | +0.007 |
| pt-en | 46.044 | 45.982 | −0.062 | −0.028 | 87.538 | 87.540 | +0.003 |
| ru-en | 30.466 | 30.448 | −0.018 | −0.009 | 82.151 | 82.136 | −0.015 |
| uk-en | 33.469 | 33.351 | −0.119 | −0.050 | 83.168 | 83.152 | −0.016 |
| zh-en | 24.384 | 25.094 | **+0.709** | +0.230 | 81.444 | 81.561 | **+0.117** |

**The engine is quality-neutral.** Eight of nine pairs land within ±0.25 BLEU and ±0.04 COMET.

**Chinese is the one real effect, and it belongs to the engine.** Both CTranslate2 configurations
beat Transformers by the same margin on zh-en — +0.73 BLEU as int8 (original Axis A), +0.71 as fp16
(here) — while differing from *each other* by +0.02. A gain common to both precisions and absent
between them is the engine.

**This agrees with the clean decomposition**, which measures the same contrast independently at
n=1,000 and reports Δengine −0.264…+0.603 against this run's −0.251…+0.709. The two runs concur
pair-by-pair on which languages move and by roughly how much, including zh-en (+0.603 vs +0.709) and
ar-en (−0.038 vs −0.251).

They did not always concur, and the reason was a defect rather than sampling noise: the *original*
decomposition read zh-en Δengine as +0.08, and an earlier revision of this report explained that
away as an underpowered subsample. It was not — that run's HF leg was scoring another system's
output. See [the two harness defects](#two-harness-defects-this-run-corrected). The lesson is
recorded because the wrong explanation was plausible enough to publish: *a discrepancy between two
measurements is a reason to inspect the artifacts, not to reconcile the numbers.*

**Quantization: a consistency check, not new evidence.** Differencing the two CT2 columns gives
fp16 − int8 of ar −0.378, bn +0.123, de +0.182, fa −0.064, fr +0.266, pt +0.117, ru +0.031,
uk −0.004, zh −0.019 — signs both ways, magnitudes in line with everything else. But this is a
*corpus-score subtraction across two separate runs*, not a paired test, and its ar-en value exceeds
the ±0.18 range the properly-paired Δquant occupied. The evidence for the quantization null result
remains the decomposition's paired bootstrap, not this arithmetic.

### Axis B — as-deployed, each branch exactly as shipped

All values below are read from the committed per-pair `axisB.{hf,ct2}.report.txt`.

| Pair | HF (`develop`) | ratio HF | **CT2 shipping** | ratio CT2 | **ΔBLEU** | Δ *before* the fix |
|---|---|---|---|---|---|---|
| ar-en | 46.19 | 0.942 | **48.80** | 0.984 | **+2.61** | −1.00 |
| bn-en | 29.86 | 0.692 | **43.89** | 0.972 | **+14.03** | +7.57 |
| de-en | 45.97 | 0.915 | **47.88** | 0.946 | **+1.91** | −0.73 |
| fa-en | 34.98 | 0.764 | **45.58** | 1.024 | **+10.60** | +7.22 |
| fr-en | 49.35 | 0.931 | **50.58** | 0.956 | **+1.23** | −0.93 |
| pt-en | 51.47 | 0.935 | **52.86** | 0.968 | **+1.39** | −1.51 |
| ru-en | 38.74 | 0.915 | **41.09** | 0.949 | **+2.35** | −0.20 |
| uk-en | 41.78 | 0.915 | **44.01** | 0.954 | **+2.23** | −0.62 |
| zh-en | 27.03 | 0.751 | **35.88** | 0.972 | **+8.85** | −10.03 |

**Every pair now favours CTranslate2, where seven of nine previously lost.** The smallest swing
against the pre-fix configuration is +2.2 BLEU (fr-en), the largest +18.9 (zh-en).

### What is actually responsible — measured, not inferred

An earlier revision of this report split the result by eye: bn/fa/zh were "dominated by
segmentation" and the remaining six were "plausibly mostly beam 4 vs greedy". **With all nine length
ratios now available that inference is testable, and it was wrong.**

BLEU factors exactly as `BP × geomean(n-gram precisions)`, so the gain splits into a *length
recovery* term and an *n-gram precision* term with no modelling assumptions:

| Pair | ΔBLEU | from length recovery | from n-gram precision | length share |
|---|---|---|---|---|
| ar-en | +2.61 | +2.13 | +0.48 | 82% |
| bn-en | +14.03 | +14.99 | −0.96 | 107% |
| de-en | +1.91 | +1.67 | +0.24 | 88% |
| fa-en | +10.60 | +12.08 | −1.48 | 114% |
| fr-en | +1.23 | +1.43 | −0.20 | 116% |
| pt-en | +1.39 | +1.97 | −0.58 | 141% |
| ru-en | +2.35 | +1.60 | +0.75 | 68% |
| uk-en | +2.23 | +1.94 | +0.29 | 87% |
| zh-en | +8.85 | +9.38 | −0.53 | 106% |

**The entire as-deployed gain is length recovery, on all nine pairs.** The precision term spans
−1.48…+0.75 BLEU and is *negative on five of nine* — translating more text slightly dilutes n-gram
precision, which is the expected signature of recovering content rather than of decoding better.
Shares above 100% are exactly that effect. (The split is order-dependent in principle; computing it
the other way moves no value by more than 0.5 BLEU and changes no sign that matters.)

Regressing ΔBLEU on Δlength-ratio across the nine pairs gives **r² = 0.971** with an intercept of
**+0.44 BLEU**. A uniform benefit from beam-4-vs-greedy would appear precisely as that intercept, so
it is bounded at well under half a BLEU point — consistent with the per-pair precision terms.

The six Latin/Cyrillic pairs were *not* a different phenomenon. Their length ratios improved too
(0.911–0.942 → 0.944–0.984); there was simply less under-generation available to recover. It is one
mechanism operating at nine different magnitudes, and the magnitude tracks how badly `develop` was
truncating that language.

That leaves attribution to segmentation rather than to beam search resting on one further link: the
splitter sweep held the decoder fixed at beam 4 and varied only chunk count, reproducing length
ratios from 0.363 to 1.031 — the full observed range — so chunk count alone is sufficient to produce
these length effects.

### The mechanism, and how it was found

Porting `develop`'s token-based splitter to the CTranslate2 branch fixed Chinese as-deployed
(length ratio 0.551 -> 0.704) but **regressed** Bangla (0.825 -> 0.666) and Persian (0.899 -> 0.756):
it imported `develop`'s own under-generation. Neither splitter was right.

A controlled sweep (`eval/sweep_splitter.sh`, 200 sentences/pair) found the actual mechanism.
**NLLB under-generates on long inputs, monotonically in chunk count** - on bn-en: 15 chunks ->
0.363, 23 -> 0.457, 42 -> 0.727, 69 -> 0.869, 238 -> 1.031. Raising the token budget makes it worse.

Setting **`SENTENCE_SPLITTER_MODE=SENTENCE`** - one sentence per chunk - fixes every pair tested.
Full scale (5,000 sentences/pair, H100) against both rejected alternatives:

| pair | char splitter *(this report's original CT2)* | token, packed | **SENTENCE (shipped)** | HF (`develop`) |
|---|---|---|---|---|
| zh-en | 17.00 / 0.551 | 25.59 / 0.704 | **35.88 / 0.972** | 27.03 / 0.751 |
| bn-en | 37.43 / 0.825 | 28.71 / 0.666 | **43.89 / 0.972** | 29.86 / 0.692 |
| fa-en | 42.20 / 0.899 | 35.05 / 0.756 | **45.58 / 1.024** | 34.98 / 0.764 |

The 200-sentence sweep predicted 0.97-1.03 and full scale delivered 0.972-1.024, so the sweep is a
sound proxy for future splitter questions — worth knowing, because the sweep runs locally in minutes
and the full pipeline takes a day on an H100.

The original sweep evidence follows.



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
2. **Adopt CTranslate2 as the engine** — **~2.4×** over Transformers on H100 (2.35–2.47× across
   nine languages) at indistinguishable quality, and more on smaller cards. Credit the gain to the
   engine, not to quantization.
3. ~~**Fix segmentation for dense scripts — in both directions.**~~ **RESOLVED, and it was the
   highest-value change in the study.** Neither splitter was the answer: the fix is
   `SENTENCE_SPLITTER_MODE=SENTENCE` (one sentence per chunk), now the shipped default. Confirmed
   at 5,000 sentences/pair on **all nine** — CTranslate2 as-deployed now beats `develop` by
   **+1.23 to +14.03 BLEU**, having lost on seven of nine before. See
   [Final results](#final-results--the-shipping-configuration-nine-pairs).
4. **Keep beam 4; never inherit `develop`'s greedy default.** It costs little on this engine. Note
   that the as-deployed margin is *not* the argument for it: the BP decomposition bounds any uniform
   beam contribution at ~+0.44 BLEU, and the +1.23…+14.03 gain is length recovery on every pair.
   Beam 4 is kept because it is standard practice and cheap here, not because this study measured a
   benefit from it.
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

- **Axis B still varies segmentation and decoding together** (greedy vs beam 4). The BP
  decomposition bounds the decoding share at ~+0.44 BLEU and attributes the rest to length recovery,
  but no run isolates beam search directly. "Segmentation causes the length recovery" rests on the
  sweep, which varied chunk count at fixed beam 4 and reproduced the full range of ratios.
- **Two harness defects were found by re-running, not by review**, and both had already been
  published: the ar-en `DIFFICULT_LANGUAGE_TOKEN_LIMIT` confound, and the zh-en HF leg scoring
  another system's output. Neither was visible in the summary tables without cross-checking the
  per-metric artifacts. Assume the same class of defect until an experiment has been reproduced from
  deleted hypothesis files.
- **The pre-fix Axis B figures are no longer backed by working-tree artifacts.** The
  `axisB.{fp16,int8}.*` files were replaced in place by `axisB.{hf,ct2}.*` when the shipping
  configuration was measured. The "Δ before the fix" column is recoverable only from git history
  (`git show ce0d5733:python/NllbTranslation/eval/pipeline-results/<pair>/axisB.int8.report.txt`).
- **Decomposition sample size** is 1,000 per pair (vs 5,000 for Axis A), so its confidence intervals
  are wider. The quantization null result is consistent across all nine pairs, which is what carries
  it, rather than any single pair's precision.
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

- **CPU-target quality.** int8 is recommended for CPU on capability grounds (fp16 is unsupported
  there), and CPU throughput is now *indicatively* measured — ~0.77 sent/s in document mode on the
  local workstation CPU under WSL2, i.e. ~1.8 h for a 5,000-sentence file. That is a viability
  signal, not a benchmark: it is not server silicon and no CPU quality run exists. Axis A's
  quantization null result carries over on the reasonable assumption that `int8_float32` on CPU
  scores like `int8_float16` on GPU, which has **not** been verified.
- **The beam-4 discriminating run.** Would isolate beam search from segmentation directly rather
  than bounding it. Largely answered by the BP decomposition, which caps any uniform beam
  contribution at ~+0.44 BLEU, and low value regardless since both changes are being kept.
- **Batched throughput for the CT2 systems.** All decomposition figures are batch 1. Batched
  CTranslate2 was spot-checked only on the RTX 5070 Ti (fp16 33.1 vs int8 32.1 sent/s).

## Artifacts

- **`pipeline-results/`** — the **final** H100 run of the shipping configuration, committed in
  `99330c06`. `SUMMARY.md`; per pair `axisA.report.txt`, `axisB.{hf,ct2}.report.txt`,
  `meta.{hf,ct2}.json`; per pair `decomp/<pair>/decomp.SUMMARY.md`,
  `decomp.{engine,quant}.report.txt` and `meta.{hf-fp16,ct2-fp16,ct2-int8}.json`. Hypothesis and
  sample files omitted for size. Every figure in this report is now read from these files, except
  the pre-fix Axis B column noted in the caveats.
  The legs were relabelled in this run: `fp16`/`int8` became `hf`/`ct2`, because the CTranslate2
  image ships fp16 on GPU and the old names had stopped describing it.
- **`results/`** — RTX 5070 Ti run (original, gitignored locally). Adds `axisA.segments.csv`
  (per-sentence + COMET) and the `hyp.*.en` files.
- **Pipeline:** `run_pipeline.sh`, `run_decomp.sh`, `tmx_sample.py`, `mt_eval.py`,
  `nllb_eval_driver.py`, `ct2_driver.py`, `convert_ct2.sh`, `summarize.py`, `decomp_report.py`.
  `sweep_splitter.sh` and `bench_split_mode.py` were written later and live on
  `prototype/nllb-ctranslate2`, not on this branch.
- **Timing** (RTX 5070 Ti, 5,000 sentences/pair): int8 23–25 min, fp16 1.3–2.1 h.
