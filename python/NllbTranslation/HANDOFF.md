# NllbTranslation / CTranslate2 — Handoff

**Purpose.** This branch will be in code review for some time, and reviewers will ask *why* things
are the way they are. Most of those answers are measurements, not preferences. This document is the
index to them, so whoever is defending the review can answer without reconstructing the reasoning.

**How to use it.** Skim [Reviewer questions](#reviewer-questions) — they are phrased the way a
reviewer will actually ask. Each answer is short, cites the evidence, and says how to reproduce it.
[Reversed decisions](#reversed-decisions) explains anything in the history that looks contradictory.
[What is not verified](#what-is-not-verified) is the honest boundary — do not defend beyond it.

**Companion documents.**

| document | what it is | where |
|---|---|---|
| `PLAN.md` | task-by-task implementation state | this branch, next to this file |
| `eval/REPORT.md` | the measurement report the decisions rest on | **`eval/nllb-mt-evaluation` branch** |
| `eval/pipeline-results/` | raw per-pair results | **`eval/nllb-mt-evaluation` branch** |
| `eval/` (here) | the harness that produces all of it | this branch — **delete before the MR**, see `PLAN.md` Phase 9 |

Read the report with:
`git show eval/nllb-mt-evaluation:python/NllbTranslation/eval/REPORT.md`

---

## What changed, in one paragraph

The component moves from PyTorch/Transformers to the **CTranslate2** inference engine. The model is
converted from `facebook/nllb-200-3.3B` at image build time, at a precision chosen by the existing
`BUILD_TYPE` arg (`gpu` → `float16`, `cpu` → `int8`). Alongside the engine change: the sentence
splitter now emits one sentence per chunk, which fixed a severe as-deployed quality defect; the
tokenizer is a swappable backend; two real component bugs were fixed; and the test suite was made
build-agnostic. Every quality claim below comes from a nine-language evaluation with significance
testing.

---

## Reviewer questions

### "Why CTranslate2 at all?"

Throughput, at no measurable quality cost. CTranslate2 at *unchanged* fp16 precision is **~2.3×**
faster than Transformers on H100 (1.7–2.4× across pairs) and **6.3×** on a consumer RTX 5070 Ti.
Quality difference is not significant on 8 of 9 pairs.

Note the speedup is **hardware-dependent** — do not quote "~6×" without naming the GPU. H100
accelerates the batched PyTorch path far more than it accelerates CTranslate2's latency-bound path.

*Evidence:* `REPORT.md` § "Engine vs quantization". Reproduce with `eval/run_decomp.sh`.

### "We shipped int8 before. Why is the GPU build fp16 now?"

Because int8 is **slower** on H100 — measured on all nine pairs, 0.80–0.98× the speed of fp16, worst
on Chinese. It is also not more accurate: with the engine held fixed, int8-vs-fp16 is
non-significant on every pair (BLEU p = 0.33–0.96, COMET p = 0.21–0.98). int8's only remaining
advantage is footprint, which is immaterial on an 80 GB card.

int8 is still used for **CPU**, where fp16 does not exist as a compute type at all.

*Evidence:* `REPORT.md` § "Throughput"; the nine-pair Δquant table.

### "Why does the build download 17 GB and convert, instead of pulling a prebuilt model?"

There is no published CTranslate2 build of NLLB-200-3.3B at fp16 — OpenNMT publishes int8 only. So
the GPU target *requires* build-time conversion. Converting both targets from one checkpoint then
buys a single provenance and one pinned revision behind every image, rather than mixing a
third-party artifact with a self-built one.

The cost is real and was accepted knowingly: ~17 GB transient per build, plus a multi-minute
conversion. Final images are 37.9 GB (gpu) / 23.2 GB (cpu), against 32 GB for the old prebuilt-int8
image and 51.2 GB for the fp16 Transformers image.

*Evidence:* `PLAN.md` Phase 1.

### "Why is `--copy_files` in the converter command? Looks incidental."

It is load-bearing. A bare `ct2-transformers-converter` run emits only `config.json`, `model.bin`
and `shared_vocabulary.json` — **no tokenizer at all**. Without it the component has nothing to
encode with. The copied `sentencepiece.bpe.model` is byte-identical (md5
`05c551ae7955b3980d5a9d044eb09d70`) to the FLORES model this component used to download separately,
so dropping that download cost nothing.

This was verified in the shipped artifact, not just between source repos.

### "Why did the sentence-splitter default change to `SENTENCE`? That affects every job."

This is the highest-impact change in the branch and the one most worth understanding.

NLLB **under-generates on long inputs**, monotonically in chunk count. Measured on bn-en: 15 chunks
→ length ratio 0.363, 23 → 0.457, 42 → 0.727, 69 → 0.869, 238 → 1.031. Packing sentences into
larger chunks loses content; one sentence per chunk does not.

Full-scale result (5,000 sentences/pair), as-deployed:

| pair | before (char splitter) | **after (SENTENCE)** | develop |
|---|---|---|---|
| zh-en | 17.00 / 0.551 | **35.88 / 0.972** | 27.03 / 0.751 |
| bn-en | 37.43 / 0.825 | **43.89 / 0.972** | 29.86 / 0.692 |
| fa-en | 42.20 / 0.899 | **45.58 / 1.024** | 34.98 / 0.764 |

Chinese as-deployed went from **−8.64 BLEU behind** develop to **+8.85 ahead**. A Latin control
(pt-en) improved slightly and did not regress.

Two points a reviewer may raise:

- *"Doesn't 6–8× more chunks cost throughput?"* No. Measured: within noise on GPU, **14% faster** on
  CPU, and ~40% more output per unit time because that output is content the old mode was losing.
- *"Wasn't the plan to port develop's token-based splitter?"* It was, and that port **regressed**
  Bangla and Persian by importing develop's own under-generation. The character-vs-token framing was
  a red herring — with one sentence per chunk the sizing unit barely matters.

*Evidence:* `REPORT.md` § "Follow-up: the as-deployed gap is chunk size". Reproduce in ~3 minutes
with `eval/sweep_splitter.sh`.

### "Why does the component skip translation when source == target?"

Because NLLB does not copy the input — it paraphrases it. `eng_Latn → eng_Latn` turns
*"This is English text"* into *"This is **an** English text"*, which is both wasted compute and a
surprising mutation of the caller's data. The check compares **full FLORES codes**, so genuine
same-language conversions such as `zho_Hans → zho_Hant` still go through the model.

### "There are properties that do nothing. Why keep them?"

`NLLB_TRANSLATION_TOKEN_SOFT_LIMIT` and `DIFFICULT_LANGUAGE_TOKEN_LIMIT` both work by overriding
`preferred_limit`, which the `SENTENCE` path never consults — so under the shipped default they are
inert. They are kept because they are **develop's** properties, they have a real effect in packed
mode (`SENTENCE_SPLITTER_MODE=DEFAULT` halves output at soft limit 512), and removing them would
diverge from develop and drop a legitimate escape hatch. Both descriptions now state the constraint.

Contrast `NLLB_LENGTH_PENALTY`, which *was* removed — see [Reversed decisions](#reversed-decisions).

### "Why are there two tokenizer backends?"

To keep the choice open without committing. `SENTENCEPIECE` is the default and is what the
evaluation measured; `HUGGINGFACE` is selectable per job via `NLLB_TOKENIZER`. They were verified
interchangeable: **20/20 byte-identical translations**, **300/300 identical token counts** (the
property that matters, since it drives chunk boundaries), and the one divergence — an em-dash
rendered `'—'` vs `<unk>` — resolves to the same token in CTranslate2. `transformers` is imported
lazily, so the default path carries no extra dependency.

### "Why does `_check_model` track the model name in an attribute?"

`ctranslate2.Translator` has no `name_or_path`. The previous code only tested `model_is_loaded`,
which is never false for a live Translator, so **a job-level `NLLB_MODEL` was silently ignored** and
the baked-in model served every request. That defect invalidated an entire decomposition run during
the evaluation before it was found. There is a regression test, and it was verified non-vacuous by
reverting the fix and confirming the test fails.

### "Why don't the tests assert exact translations any more?"

Because the two build targets legitimately produce different text: gpu (`float16`) and cpu
(`int8_float32`) agree on only 24 of 30 sentences. Divergence tracks **lexical ambiguity, not
length** — a 27-character sentence diverges (`canhões` → "cannons" vs "guns") — so it cannot be
dodged by choosing simpler inputs, and selecting inputs for stability would select for trivial ones.

28 of 31 exact-output assertions became structural, each with a one-line comment stating what the
test actually verifies. Exact output is still pinned in `test_long_spanish`, which is gated behind
`RUN_DEEP_TESTS` and baselined on the gpu build.

### "How do I know the tests actually test anything?"

Where a test guards a specific bug, it was verified by breaking the fix and confirming the test
fails:

- reverting `_check_model` → both `NLLB_MODEL` tests fail
- drifting a `JobConfig` default → the descriptor-drift test fails (`4 != 8`)
- removing a kwarg from `translate_batch` → the plumbing test errors

This matters here: an early beam-size check passed vacuously because the test sentence was too easy
to distinguish greedy from beam search. **Easy input does not discriminate.**

---

## Reversed decisions

History contains reversals. They are deliberate and evidence-driven; this is why.

| decision | first taken | reversed to | why |
|---|---|---|---|
| GPU precision | int8_float16 | **float16** | H100 measurement: int8 slower on all 9 pairs |
| model source | prebuilt OpenNMT int8 | **convert from facebook** | no fp16 prebuilt exists; single provenance |
| `NLLB_LENGTH_PENALTY` | added | **removed** | added to counter under-generation; measured identical output at 1.5, so the rationale evaporated |
| splitter fix | port develop's token splitter | **`SENTENCE` mode** | the port regressed bn/fa; chunk *count* was the real variable |
| `OUTPUT_MERGE_WITH_PREVIOUS_TASK` | "keep, develop lacks it" | **dropped** | it is *retired*, not missing — `cb351e4a` renamed it to `IS_ANNOTATOR` repo-wide |

---

## What is *not* verified

Do not defend these as measured; they are not.

- **The full-scale as-deployed result covers bn/fa/zh only.** The five Latin/Cyrillic pairs have not
  been re-run at full scale under `SENTENCE` mode. A pt-en control at N=200 improved slightly, so
  regression is unlikely — but "CT2 beats develop as-deployed" currently rests on three languages.
- **The ar-en engine outlier is unexplained.** It is the only pair with a large Δengine (+1.91 BLEU)
  and the slowest HF throughput. An earlier claim that this was disproved as a chunking artifact was
  itself wrong — see the trap below.
- **Quantization neutrality is per-language for Axis A, but the decomposition is n=1,000 per pair**
  versus 5,000 for Axis A. The consistency across nine pairs carries it, not any single pair.
- **CPU throughput is measured on 4 container cores** at CTranslate2's default threading. A larger
  host will differ; `NLLB_INTER_THREADS`/`NLLB_INTRA_THREADS` exist but are unmeasured.

---

## Traps in this codebase

Each of these cost real time. They share a shape: **absence of change reads as confirmation.**

1. **The descriptor's `defaultValue` is the operative default, not `JobConfig`'s.** The executor
   reads the descriptor and passes every property explicitly, so a `JobConfig` default is only a
   fallback for callers that omit it. Changing a default means changing **both**. Symptom: a change
   appears to have no effect because the image's descriptor still supplies the old value.
2. **`run_pipeline.sh` and `run_decomp.sh` skip regeneration** when the hypothesis file is already
   complete. An experiment re-run without deleting `hyp.*.en` first silently re-scores the old text
   and "produces identical results" regardless of what you changed. This produced a wrong conclusion
   about the ar-en outlier that survived two documents.
3. **`RUN_TESTS=true` runs in the build stage**, which has no GPU — so it exercises the CPU
   inference path for *both* build types. The GPU path needs a separate `--gpus` run.
4. **Easy input does not discriminate decode settings.** Beam 1 and beam 4 agree on short sentences.
   Any test asserting a decode property "does something" needs hard input or it passes vacuously.

---

## Reproducing the evidence

```bash
# the full pipeline, one pair (hours)
./eval/run_pipeline.sh zh-en

# splitter behaviour, ~3 minutes per pair -- predicted full-scale ratios to within 0.01
COMPONENT_SRC=../nllb_component ./eval/sweep_splitter.sh          # PAIR=, N= to vary

# throughput, in-process so model load is excluded
docker run --rm --gpus '"device=0"' -v "$PWD/eval":/eval:ro -v "$PWD/eval/results":/results:ro \
  --entrypoint /opt/mpf/plugin-venv/bin/python IMAGE /eval/bench_split_mode.py --pair bn-en -n 200

# the test suite, against a built image
docker run --rm --gpus '"device=0"' -v "$PWD":/component:ro --entrypoint bash IMAGE \
  -c 'cd /component/tests && /opt/mpf/plugin-venv/bin/python -m unittest test_nllb_translation'
```

`eval/` needs the TMX corpora under `eval/tmx/` and a scoring venv (`eval/setup_venv.sh`); neither
is in git. `eval/README.md` covers setup.

---

## Open work

`PLAN.md` is authoritative. In short: Phases 0–8 are complete; **Phase 9 (pre-merge cleanup) is
not started** and includes deleting `eval/` from this branch. Optional follow-ups are task 8.5b (the
ar-en outlier) and 8.6 (cache `TextSplitterModel`, a small optimisation).
