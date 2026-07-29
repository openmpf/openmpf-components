# NLLB-200-3.3B: fp16 vs int8 Translation Quality Evaluation

**Question:** Does int8 quantization (`OpenNMT/nllb-200-3.3B-ct2-int8`, CTranslate2) degrade
translation quality vs the fp16 original (`facebook/nllb-200-3.3B`, PyTorch), as deployed in the
OpenMPF NllbTranslation component?

**Data:** OPUS TED2020 → English. Nine source languages, 5,000 sentence pairs each (fixed random
sample, seed 42), gold sentence-aligned references from TMX.

| Pair | Source | NLLB code | Corpus (TU) |
|---|---|---|---|
| pt-en | Portuguese | por_Latn | 320k |
| ar-en | Arabic (MSA) | arb_Arab | 398k |
| zh-en | Mandarin (Simplified) | zho_Hans | 393k |
| bn-en | Bangla | ben_Beng | 10k |
| de-en | German | deu_Latn | 289k |
| en-fr | French | fra_Latn | 400k |
| en-uk | Ukrainian | ukr_Cyrl | 203k |
| en-ru | Russian | rus_Cyrl | 380k |
| en-fa | Persian (Western) | pes_Arab | 297k |

**Setup:** 2026-07-25 → 2026-07-28, RTX 5070 Ti (16 GB). sacrebleu 2.6.0; COMET
`Unbabel/wmt22-comet-da`. Three experiments:

- **Axis A — intrinsic** (all 9 pairs): per-sentence, beam 4 on both, sentence-splitting
  neutralized. Isolates the deployed-model variable.
- **Axis B — as-deployed** (pt/ar/zh): whole file through each branch's own shipped pipeline
  (document-level).
- **Engine-vs-quantization decomposition** (pt-en, 1,000 sentences): splits Axis A's single
  contrast into its two confounded halves — inference *engine* and *numeric precision*.

---

## Bottom line

1. **Intrinsic quality: int8 is equivalent to fp16 in all nine languages.** Every per-sentence
   delta is ≤0.34 BLEU and ≤0.15 COMET on 0–100 scales, and they point in *both* directions
   (int8 significantly ahead on Chinese, significantly but trivially behind on pt/de/fr, and
   statistically tied on the other five). No language shows meaningful degradation.
2. **The residual sub-BLEU wobble is the *engine*, not the quantization.** Holding the engine
   fixed, int8-vs-fp16 is statistically indistinguishable — ΔBLEU −0.04 (p=0.83), ΔchrF +0.04
   (p=0.80), ΔCOMET −0.00 (p=0.96). Quantization costs nothing.
3. **The ~6× speedup is also the *engine*, not int8.** CTranslate2-fp16 is 6.3× faster than
   Transformers-fp16 at identical quality; int8 adds only ~5% on top of that. int8's real payoff
   is **memory** (~⅓), not throughput.
4. **As-deployed: int8 lags, and the gap scales with script density** — modest for pt/ar
   (~1.4–2 BLEU), **catastrophic for Chinese (−8.6 BLEU)**. This is **not quantization**; it is the
   int8 branch's older *character-based* sentence splitter under-generating on dense text. It is
   fixable, and fixing it is a priority.

---

## Axis A — intrinsic quality (controlled, the fair comparison)

Per-sentence, beam 4 on both, splitter neutralized. Δ = int8 − fp16. Significance = paired
bootstrap, 1,000 resamples. 5,000 segments per pair.

| Pair | BLEU fp16 | BLEU int8 | ΔBLEU (sig) | ΔchrF (sig) | COMET fp16 | COMET int8 | ΔCOMET (sig) | Identical outputs |
|---|---|---|---|---|---|---|---|---|
| pt-en | 46.04 | 45.87 | −0.18 (p=0.048) | −0.08 (n.s.) | 87.54 | 87.50 | −0.04 (n.s.) | 78% |
| ar-en | 40.71 | 40.83 | +0.12 (n.s.) | +0.05 (n.s.) | 84.88 | 84.90 | +0.02 (n.s.) | 71% |
| zh-en | 24.38 | 25.11 | **+0.73** (p<0.001) | +0.19 (p=0.07) | 81.44 | 81.59 | **+0.15** (p<0.001) | 58% |
| bn-en | 33.24 | 33.15 | −0.10 (n.s.) | −0.08 (n.s.) | 85.75 | 85.74 | −0.01 (n.s.) | 66% |
| de-en | 39.04 | 38.70 | **−0.34** (p<0.001) | −0.15 (p=0.026) | 85.94 | 85.87 | −0.07 (p=0.016) | 75% |
| en-fr | 42.61 | 42.32 | **−0.29** (p<0.001) | −0.18 (p=0.002) | 86.13 | 86.09 | −0.04 (n.s.) | 75% |
| en-uk | 33.47 | 33.35 | −0.11 (n.s.) | −0.02 (n.s.) | 83.17 | 83.15 | −0.02 (n.s.) | 70% |
| en-ru | 30.47 | 30.42 | −0.05 (n.s.) | +0.00 (n.s.) | 82.15 | 82.11 | −0.04 (n.s.) | 68% |
| en-fa | 36.17 | 36.19 | +0.02 (n.s.) | −0.02 (n.s.) | 84.90 | 84.93 | +0.02 (n.s.) | 70% |

(TER, where the length guard allowed it: zh 70.14→66.51 **int8 better by 3.6**, bn 56.46→56.27 and
fa 52.75→52.68 int8 better, fr 43.89→44.10 and uk 54.00→54.09 fp16 marginally better.
pt/ar/de/ru skipped — a segment exceeded the TER length guard.)

**Reading it.** The largest delta anywhere is 0.34 BLEU (de-en) — well inside the range where two
systems are considered equivalent, and an order of magnitude below the between-language spread
(BLEU 24–46). The deltas also *disagree in sign*: int8 wins zh clearly (+0.73 BLEU, +0.15 COMET,
−3.6 TER — all significant), loses pt/de/fr by ~0.2–0.3, and is statistically tied on
ar/bn/uk/ru/fa. With 5,000 paired segments the bootstrap resolves differences this small as
"significant," but statistical significance here is not practical significance: a 0.3-BLEU /
0.07-COMET shift is not visible in output quality.

The two languages added last (Russian, Persian) are the strongest single data points for
equivalence: **every metric is non-significant on both**, with ΔBLEU of −0.05 and +0.02 and
p-values from 0.21 to 1.00. They also widen the script coverage to a second Cyrillic and a second
Arabic-script language without changing the picture.

The identical-output rate (58–78%) tracks language difficulty — the harder the language, the more
the two decoders diverge — but where they diverge, quality is a wash.

Note that Axis A still compares **two things at once**: Transformers-fp16 vs CTranslate2-int8.
Engine and precision are confounded. The decomposition below separates them.

## Engine vs quantization — decomposing the Axis A contrast

Axis A's "fp16 vs int8" bundles two independent changes: the *inference engine* (HuggingFace
Transformers → CTranslate2) and the *numeric precision* (fp16 → int8). These are orthogonal —
CTranslate2 runs fp16 perfectly well — so a third system, **CTranslate2-fp16**, splits the
contrast cleanly. Run on an independent 1,000-sentence pt-en sample, all three per-sentence at
beam 4.

Δengine = CT2-fp16 − HF-fp16 (same precision, different engine).
Δquant = CT2-int8 − CT2-fp16 (same engine, different precision).

| Metric | HF-fp16 | CT2-fp16 | CT2-int8 | Δengine (sig) | Δquant (sig) |
|---|---|---|---|---|---|
| BLEU | 45.06 | 44.92 | 44.88 | −0.14 (p=0.46, n.s.) | −0.04 (p=0.83, n.s.) |
| chrF | 66.32 | 66.23 | 66.27 | −0.09 (p=0.60, n.s.) | +0.04 (p=0.80, n.s.) |
| chrF++ | 65.01 | 64.92 | 64.94 | −0.09 | +0.03 |
| COMET | 87.35 | 87.37 | 87.37 | +0.02 (p=0.60, n.s.) | −0.00 (p=0.96, n.s.) |

**Quantization is a no-op for quality.** With the engine held constant, every int8-vs-fp16 delta is
within noise (p ≥ 0.80 on all three tested metrics), even though the two systems genuinely produce
different text (214/1000 outputs differ). Whatever small movement Axis A shows is attributable to
the engine's beam-search internals, not to the int8 weights. For pt-en the two halves also sum
consistently with the Axis A result (−0.14 + −0.04 ≈ −0.18 observed).

### Throughput (sentences/sec, single-sentence latency, batch 1)

| System | sent/s | speedup vs HF-fp16 |
|---|---|---|
| HF-fp16 (Transformers) | 0.63 | 1.0× |
| CT2-fp16 (CTranslate2) | 3.96 | **6.3×** |
| CT2-int8 (CTranslate2) | 4.15 | **6.6×** |

**The speedup is the engine.** Moving Transformers → CTranslate2 at *unchanged* fp16 precision
already buys 6.3×; int8 adds a further ~5%. Batched, the two CTranslate2 configurations are
effectively tied (33.1 vs 32.1 sent/s measured) — on this GPU int8 does not unlock a tensor-core
win over fp16. int8's genuine advantage is **memory footprint** (~⅓ of fp16), which matters for
model co-residency and for smaller cards, not raw speed.

This reframes the throughput claim: the win we have been attributing to int8 is really the win
from leaving the PyTorch/Transformers generation loop. **CT2-fp16 is a viable "fast fp16"** — same
6× speedup, no quantization at all — if quantization ever needs to be taken off the table.

For reference, the main pipeline's own timings (fp16 batched at 16 vs int8 per-line) show int8
**3.3–4.7× faster** per pair — int8 3.32–3.66 sent/s, fp16 0.74–1.11 sent/s — across the seven
pairs whose run metadata records per-second throughput (pt, bn, de, fr, uk, ru, fa). int8 wins by
that margin *despite* running unbatched, consistent with the controlled figures above.

## Axis B — as-deployed (each branch exactly as shipped)

Whole 5,000-sentence file as one document through each branch's own pipeline (fp16: greedy +
`sat-3l-sm` token-based splitter; int8: beam 4 + `wtp-bert-mini` **character-based** splitter).
Run on pt/ar/zh only — the other six pairs were Axis A only (`RUN_AXIS_B=0`).

| Pair | BLEU fp16 | BLEU int8 | ΔBLEU | length ratio fp16 | length ratio int8 |
|---|---|---|---|---|---|
| pt-en † | 51.53 | 50.14 | −1.39 | 0.940 | 0.903 |
| ar-en | 46.19 | 44.24 | −1.95 | 0.942 | 0.892 |
| zh-en | 27.03 | **18.39** | **−8.64** | 0.751 | **0.570** |

† pt-en Axis B was measured on the earlier 5,000-sentence Moses-derived pt sample, not the
TMX-derived sample used for the Axis A row above. The comparison is internally paired
(same input to both branches), so the Δ stands.

**Reading it — this gap is a splitter defect, not quantization.** The tell is the **length ratio**:
int8's as-deployed output is systematically short, and for Chinese it collapses to **57% of the
reference length**, triggering a severe brevity penalty. Root cause: the int8 branch splits by
*character count* (360 chars). Chinese has no spaces and is character-dense, so 360 chars is a huge
span → few, very long chunks → NLLB under-translates/truncates them. `develop` splits by *token
count* (130-token target with `sat-3l-sm`), producing right-sized chunks and near-complete output.
Axis A proves the int8 *model* does not under-generate per sentence (it matches/beats fp16); the
under-generation appears only in document mode, pinning the blame on the splitter.

---

## Recommendation

1. **Ship int8.** Confirmed across nine languages and five scripts: intrinsic quality is equivalent
   to fp16, for ~6× throughput and ~⅓ the memory. Strong, multi-language, significance-tested
   backing, with the quantization variable now isolated and shown to be a no-op.
2. **Port `develop`'s token-based splitter (`sat-3l-sm` + `NLLB_TRANSLATION_TOKEN_LIMIT` logic) to
   the ctranslate2 branch — a priority.** It is the *entire* as-deployed gap, and it is severe
   for CJK/dense scripts. **As shipped today, the int8 as-deployed pipeline is not fit for Chinese.**
3. **Interim mitigation** (before the port): drop `SENTENCE_SPLITTER_CHAR_COUNT` well below 360 for
   dense-script sources, or feed pre-segmented input.
4. **Fix `NLLB_MODEL` handling on the ctranslate2 branch (component bug, found during this work).**
   The component loads `DEFAULT_NLLB_MODEL` in `__init__` and `_check_model` only reloads
   `if not self._model.model_is_loaded` — so a job-level `NLLB_MODEL` property is silently
   **ignored** and the baked-in model is used regardless. Any deployment relying on per-job model
   selection is not getting it. (Worked around here with a standalone `ct2_driver.py`.)
5. **Credit the speedup correctly in downstream docs.** It is CTranslate2, not int8. If int8 ever
   becomes contentious, CT2-fp16 delivers the same ~6× at bit-exact fp16 precision.

## Caveats

- **Cross-engine decoding:** Axis A harmonizes beam *size* (4) but HF's `generate` and
  CTranslate2's `translate_batch` differ in beam-search internals (length penalty, normalization).
  The decomposition attributes the small Axis A deltas to exactly this, and shows the effect is not
  statistically significant even at the engine level on pt-en (n=1,000).
- **Decomposition scope:** run on pt-en only, n=1,000. The quantization no-op result is
  well-supported there (p≥0.80 on every metric) but has not been replicated per language; the
  nine-language Axis A evidence is what covers language breadth.
- **Bangla corpus is small:** the bn-en TMX holds only 10,260 units, so the 5,000-sentence sample
  is ~49% of the available corpus. The paired comparison is unaffected, but the bn sample is less
  independent of the corpus than the others.
- **Per-language difficulty:** absolute scores vary by language (zh hardest at BLEU ~25; that's the
  language, not the engine). Arabic is NLLB's flagged "difficult" language; its special split limit
  was disabled in Axis A so both branches segment it identically.
- TED2020 references are loose crowd translations (caps absolute BLEU); possible pretraining
  contamination inflates absolute scores. Both affect fp16 and int8 equally — the deltas stand.
- **pt-en Axis A was re-run** on a TMX-derived sample (BLEU 46.04/45.87, Δ−0.18) replacing an
  earlier Moses-derived sample (46.47/46.23, Δ−0.24). Both agree: negligible, int8 marginally
  behind.
- **en-ru's fp16 leg was resumed**, not run in one pass: the first attempt died at 3,992/5,000 on a
  CUDA "unknown error" and `--resume` completed the remaining 1,008 lines. The hypotheses are
  complete and the scoring is over all 5,000 segments, but the recorded fp16 throughput for that
  pair covers only the resumed portion and is excluded from the timing ranges below.
- **en-fa needed a language-code fix first.** The initial run was skipped by the fail-fast smoke
  test because the configured code was `per_Arab` — `per` is the ISO 639-2/B bibliographic
  abbreviation, and FLORES-200 uses ISO 639-3. Verified against the NLLB tokenizer:
  `per_Arab` → UNK (id 3), **`pes_Arab` → id 256053**. Corrected to `pes` and re-run clean.

## Not yet run

- **Axis B for bn/de/fr/uk/ru/fa** — those runs used `RUN_AXIS_B=0`. The splitter finding is
  already established on pt/ar/zh; extending it is optional, though a dense-script pair would
  strengthen the zh result.
- **Decomposition for languages other than pt-en** (`run_decomp.sh`, needs the converted
  `float16` + `int8_float16` models from `convert_ct2.sh`).

## Artifacts

`results/SUMMARY.md` (combined table). Per pair in `results/<pair>/`: `hyp.{fp16,int8}.en`,
`axisA.report.txt`, `axisA.segments.csv` (per-sentence + COMET), `meta.*.json`; plus
`asdeployed.{fp16,int8}.json` / `axisB.*.report.txt` for pt/ar/zh. Decomposition in
`results/decomp/pt-en/`: `decomp.SUMMARY.md`, `decomp.{engine,quant}.report.txt`,
`hyp.{hf-fp16,ct2-fp16,ct2-int8}.en`. Pipeline: `run_pipeline.sh` and `run_decomp.sh` (+
`tmx_sample.py`, `mt_eval.py`, `nllb_eval_driver.py`, `ct2_driver.py`, `convert_ct2.sh`,
`summarize.py`, `decomp_report.py`). Timing: int8 23–25 min/pair, fp16 1.3–2.1 h/pair.
