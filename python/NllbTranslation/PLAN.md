# NllbTranslation: CTranslate2 — Implementation Plan

**Working branch:** `prototype/nllb-ctranslate2`, drawing from `develop`.
**Goal:** ship the NllbTranslation component on the **CTranslate2** inference engine, with the model
converted from a **single source — `facebook/nllb-200-3.3B`** — at a precision chosen by the
existing `BUILD_TYPE` docker arg: **`gpu` → `float16`, `cpu` → `int8`**. Keep the **tokenizer
backend swappable** (FLORES SentencePiece today, HuggingFace `AutoTokenizer` as a first-class
alternative).

**Primary deployment target is H100-class GPU;** CPU is a supported secondary target.

> ## ⚠️ `eval/` is a prototype enabler — remove it before the merge request
>
> This branch is deliberately named with a **`prototype/`** prefix. The `eval/` directory carries the
> machine-translation evaluation harness (`run_pipeline.sh`, `run_decomp.sh`, `mt_eval.py`,
> `nllb_eval_driver.py`, `ct2_driver.py`, and supporting scripts). It is here to make the plan
> *executable* — Phases 2.5, 7 and 8 all invoke these scripts, and without them the validation steps
> cannot be run from this checkout.
>
> **It is development tooling, not product code.** It benchmarks two OpenMPF *images* against each
> other, depends on TMX corpora and a separate scoring venv, and has no role at runtime.
>
> **Before opening a merge request to ship the component, delete `python/NllbTranslation/eval/`.**
> Nothing under `nllb_component/`, `plugin-files/`, `tests/` or the `Dockerfile` references it, so
> removal is a clean `git rm -r`. The evaluation record — including `REPORT.md` and the raw
> `pipeline-results/` — is preserved independently on the **`eval/nllb-mt-evaluation`** branch, so
> deleting it here loses nothing.

**Evidence base:** `REPORT.md` and the H100 runs in `pipeline-results/`, kept on the
**`eval/nllb-mt-evaluation`** branch (not copied here, to avoid a second copy that drifts). Read with
`git show eval/nllb-mt-evaluation:python/NllbTranslation/eval/REPORT.md`.
Five findings drive this plan:

- **Quantization is a no-op for quality — established across all 9 languages.** Holding the engine
  fixed, Δ(int8 − fp16) is non-significant on every pair: BLEU p = 0.33–0.96, COMET p = 0.21–0.98,
  signs mixed. Precision can therefore be chosen purely on speed/memory grounds.
- **On H100, int8 is *slower* than fp16 — on all 9 pairs** (CT2-int8 is 0.80–0.98× CT2-fp16, ~8.5%
  slower on average, 20% slower on Chinese). int8's only remaining advantage is footprint, which is
  irrelevant on an 80 GB card. **This is why the GPU build uses fp16.**
- **The engine win is real but hardware-dependent:** CT2-fp16 vs HF-fp16 is ~2.3× on H100
  (1.7–2.4× on eight of nine pairs; ar-en is an unexplained outlier at 4.1×) versus 6.3× on a
  consumer RTX 5070 Ti. Do not quote "~6×" unqualified.
- **Axis A is hardware-independent.** The H100 run reproduced the RTX 5070 Ti Axis A scores to three
  decimals on all 9 pairs — the quality conclusions do not depend on the GPU.
- **The as-deployed splitter picture is bimodal, and `develop`'s splitter is not uniformly better.**
  int8's character splitter is catastrophic on Chinese (−10.0 BLEU, length ratio 0.551), but
  `develop`'s own as-deployed pipeline *under-generates* on Bengali (0.692) and Persian (0.764),
  where int8 wins by ~7 BLEU. Axis B confounds splitter with decoding (`develop` = greedy, CT2 =
  beam 4). See Phase 3 and Phase 5.

**Branch topology.** The CT2 lineage forked from `develop` at `9adca039` (Feat/py3.12).
**The component did not exist at that merge base** — `develop` added it in `23925349` (PR #384) and
the CT2 branch added it independently in `9ac2ae56`. The two lineages therefore shared *no* history
for these files, which is why the sync merge produced `add/add` on all 8 of them with no common
ancestor to arbitrate: nothing auto-resolved, and every divergence was a hand decision.

---

## Status

**Phase 0 is done** (merge commit `8d300300`). Because `develop`'s config and its splitter live in
the same conflict hunk, the merge necessarily absorbed **most of Phase 3** as well — the two could
not be separated. The plan's original "Phase 0 first, in its own commit, before any feature work"
did not survive contact with the add/add reality.

What the merge landed, and what it left:

| Phase | State after `8d300300` |
|---|---|
| 0 — branch hygiene | **done** |
| 1 — model packaging | **done and built** — both `BUILD_TYPE` images verified |
| 2 — tokenizer abstraction | **done**; default unchanged (SENTENCEPIECE), full-scale A/B still owed |
| 3 — token-based splitter | **mostly landed**; validation outstanding |
| 4 — model lifecycle bug | still broken; `_current_model_name` is now tracked but unused |
| 5 — decode/batching params | 5.1 partly done, 5.3 done |
| 6 — descriptor/properties | **done** via the merge |
| 7 — tests | **now blocking — suite is red, `RUN_TESTS` fails the build** |
| 8 — validation | not started |

---

## Phase 0 — Branch hygiene ✅

- [x] **0.1 Sync with `develop`** — merged, not rebased (`8d300300`). Resolution policy held: take
      `develop` for splitter/config logic, keep CT2 for the inference call path. Per-file outcomes:
      - `nllb_translation_component.py` — `develop`'s structure and `JobConfig`, CT2 layer
        transplanted (`ctranslate2.Translator`, SentencePiece, `_resolve_device`, `translate_batch`).
      - `nllb_utils.py` — `develop`'s table (lowercase script keys), `.lower()` normalisation and
        `DetectionException` messaging, plus the CT2 branch's `_flores_to_wtpsplit_iso_639_1`
        (all 128 entries verified) and `get_normalized_iso`. **Fixes** the CT2 branch's
        case-sensitivity: `get_code("POR","LATN")` now resolves.
      - `descriptor.json`, `README.md`, `__init__.py`, `tests/` — `develop`'s.
      - `Dockerfile` — `develop`'s structure plus the CT2 model/SPM downloads and cuBLAS install.
- [x] **0.2 Fix `setup.cfg`** — version 10.0, `mpf_component_{api,util}>=10.0`, bogus
      `NllbTranslation` self-dependency removed, `ctranslate2`/`sentencepiece` kept. `transformers`
      still **not** declared; add it if the HF tokenizer backend ships enabled (Phase 2).
- [x] **0.3 Descriptor version** — 10.0 via `develop`'s descriptor.

## Phase 1 — Model packaging (single source, precision keyed to `BUILD_TYPE`)

**Decision: convert from `facebook/nllb-200-3.3B` for both targets, and let the existing
`BUILD_TYPE` arg pick the precision.** The prebuilt `OpenNMT/nllb-200-3.3B-ct2-int8` is dropped
entirely, so there is exactly one upstream checkpoint and one pinned revision behind every image.

| `BUILD_TYPE` | `--quantization` | resolves to | model size |
|---|---|---|---|
| `gpu` (default) | `float16` | `float16` | 6.7 GB |
| `cpu` | `int8` | `int8_float32` | 3.36 GB |

Why this split, measured not assumed (`ctranslate2` 4.8.1):

- **CPU supports only `{float32, int8_float32, int8}`** — no `float16` at all. A float16-converted
  model loaded on CPU is silently up-converted to `float32` ("the target device or backend do not
  support efficient float16 computation"), which forfeits the size win and buys nothing. Shipping the
  GPU artifact to a CPU target is therefore actively wrong, not merely suboptimal.
- **`int8_float16` on CPU is a hard error**, not a fallback:
  `ValueError: Requested int8_float16 compute type, but the target device or backend do not support
  efficient int8_float16 computation.` So the CPU build must convert with `int8`.
- On GPU, fp16 is both faster than int8 (all 9 pairs) and quality-equivalent — see the evidence base.

- [x] **1.1 Replace the `download_model` stage with a `convert_model` stage** keyed to `BUILD_TYPE`:

      ARG BUILD_TYPE=gpu
      ARG SRC_MODEL=facebook/nllb-200-3.3B
      ARG SRC_REVISION=1a07f7d195896b2114afcb79b7b57ab512e7b43e
      RUN pip install -U "huggingface_hub[cli]" ctranslate2 transformers sentencepiece
      RUN if [ "$BUILD_TYPE" = "cpu" ]; then Q=int8; else Q=float16; fi; \
          ct2-transformers-converter --model $SRC_MODEL --revision $SRC_REVISION \
            --output_dir /models/nllb-200-3.3B-ct2 \
            --quantization "$Q" --low_cpu_mem_usage \
            --copy_files sentencepiece.bpe.model tokenizer.json \
                         tokenizer_config.json special_tokens_map.json

- [x] **1.2 Use a *stable* output directory name** (`/models/nllb-200-3.3B-ct2`) rather than encoding
      the quantization in the path. `DEFAULT_NLLB_MODEL` then does not vary by build type and the
      component needs no build-type awareness. The deployed precision is still discoverable at
      runtime via the `compute_type` log in 1.5 — which is the more reliable place for it anyway.
      (This reverses an earlier note that suggested putting the quantization in the directory name.)
- [x] **1.3 `--copy_files` is now mandatory on *both* paths, and is load-bearing.** A bare
      `ct2-transformers-converter` run emits only `config.json`, `model.bin`, and
      `shared_vocabulary.json`. With the OpenNMT download gone, `--copy_files` is the **only** source
      of the SentencePiece model *and* the HF tokenizer files. Omit it and both tokenizer backends
      break.
- [x] **1.4 Drop the separate FLORES SPM download — it is the same file.** Verified byte-identical:
      `facebook/nllb-200-3.3B/sentencepiece.bpe.model` and
      `OpenNMT/nllb-200-onmt/flores200_sacrebleu_tokenizer_spm.model` both md5
      `05c551ae7955b3980d5a9d044eb09d70`. Copying `sentencepiece.bpe.model` from the source
      checkpoint reproduces exactly the tokenizer the component uses today, from a single provenance.
      **Code change required:** `SP_MODEL_PATH` at `nllb_translation_component.py:50` currently points
      at `/models/OpenNMT/flores200_sacrebleu_tokenizer_spm.model` and must move to the converted
      model dir.
- [x] **1.5 Log and assert the resolved compute type at load.** *(Implemented as a WARNING, not an exception — a mismatch means slower/heavier, not wrong, and failing a serviceable deployment is the worse outcome. The Dockerfile sets `NLLB_EXPECTED_COMPUTE_TYPE` per `BUILD_TYPE`; the component compares against it.)* Log `Translator.compute_type` after
      loading, and fail loudly if it disagrees with what the build intended. This matters more now
      that precision is build-time-conditional: it is the only runtime evidence of which artifact is
      deployed, and it catches both the silent float16→float32 CPU up-conversion and the
      `int8_float32` GPU trap. `eval/ct2_driver.py` already records `actual_compute_type` — reuse it.
- [x] **1.6 Note the `int8` warning is GPU-specific and inverts on CPU.** *(Recorded as a comment on the conversion `RUN` so it is visible where someone would 'fix' it.)* On **GPU**, bare `int8`
      resolves to `int8_float32`: accurate but with no tensor-core speedup — a trap (it cost us a
      whole decomposition run). On **CPU**, `int8_float32` is the *only* int8 mode available and is
      exactly what we want. Same flag, opposite verdict; do not "fix" the CPU build to
      `int8_float16`.
- [x] **1.7 Update `DEFAULT_NLLB_MODEL`** at `nllb_translation_component.py:49` from
      `'OpenNMT/nllb-200-3.3B-ct2-int8'` to `'nllb-200-3.3B-ct2'`.
- [x] **1.8 Size and build cost — measured.** Both images built successfully.

      | | `BUILD_TYPE=gpu` | `BUILD_TYPE=cpu` |
      |---|---|---|
      | image | **37.9 GB** | **23.2 GB** |
      | `model.bin` | 6.69 GB (float16) | 3.36 GB (int8) |
      | resolved `compute_type` | `float16` | `int8_float32` |

      For reference the previous prebuilt-int8 image was 32 GB and the fp16
      HF-checkpoint image 51.2 GB, so the GPU image is ~6 GB larger than what it
      replaces and the CPU image is substantially smaller (it skips the CUDA venv).
      Both resolved compute types matched `NLLB_EXPECTED_COMPUTE_TYPE` exactly, and
      the copied `sentencepiece.bpe.model` in each artifact carries md5
      `05c551ae7955b3980d5a9d044eb09d70` — the same file the component used to
      download separately. The 17 GB transient checkpoint did not cause a build
      failure on this host; CI disk headroom is still worth confirming separately.

- [x] **1.9 CPU target expectations — measured, and it is slow.** On 4 container
      cores the CPU image runs **0.215 sent/s** (~4.7 s/sentence) against **1.88
      sent/s** for the GPU image on an RTX 5070 Ti — roughly **9× slower**, and the
      GPU figure is itself batch-1 on a consumer card. Extrapolated, a
      5,000-sentence job is ~6.5 hours on CPU. **Treat the CPU build as a
      portability/functionality option, not a throughput one**, and say so in any
      deployment guidance.
      - [ ] **1.9a Expose CTranslate2 threading** (`inter_threads` / `intra_threads`)
            as properties. It matters far more on CPU than GPU, and the measurement
            above was taken at CTranslate2's defaults on only 4 visible cores, so
            there is likely headroom on a larger host. Folds naturally into Phase 5.1.

**Consequence for Phase 7:** the two build targets **produce different
translations**. On a 10-sentence sample, CPU (`int8_float32`) and GPU (`float16`)
agreed on 8 and diverged on 2 (e.g. *"So they brought the guns."* vs *"...the
cannons."*). This is the same quantization-level divergence the evaluation
measured as quality-neutral — but it means any expected-string test suite is
pinned to whichever `BUILD_TYPE` generated it. See open question 5.

## Phase 2 — Tokenizer abstraction (keep options open)

### What the investigation found

Probed inside the images (transformers 5.13.0, CPU):

| Question | Result |
|---|---|
| Does `AutoTokenizer` load from a CT2 model dir? | **Yes** — `NllbTokenizer`, vocab 256,204, given the copied tokenizer files |
| Are SPM pieces compatible with the CT2 vocabulary? | **Yes — 0 of 256,000 SPM pieces missing** from CT2's `shared_vocabulary.json` |
| Do the ID spaces agree? | **Yes** — `por_Latn` = 256141 in both HF and CT2 vocab order |
| Do the two tokenizers produce identical token sequences? | **Mostly** — 3/5 probe sentences identical |

The two divergences, both benign:

1. **Unknown characters.** `—` (em-dash) is in *neither* vocabulary. SentencePiece emits the raw
   character as a surface piece; HF emits `<unk>`. CT2 resolves the SPM surface form to `<unk>`
   anyway, so the model sees the same thing. Verified end-to-end: feeding the SPM piece list
   through CT2 translated correctly ("I paid EUR 3.50 for coffee, for example.").
2. **Trailing whitespace.** HF appends an extra `▁` token that SPM does not. Cosmetic.

**Conclusion: the backends are functionally interchangeable.** The swap is a low-risk design
choice, not a gamble — but it must still be A/B'd (task 2.5) before changing the default.

**Both backends now load from the same directory.** Since `--copy_files` brings
`sentencepiece.bpe.model` *and* the HF tokenizer files into the converted model dir (task 1.3), and
that SPM is byte-identical to the FLORES SPM used today (task 1.4), the SentencePiece and
HuggingFace backends read from one location with no separate download. This removes the
`/models/OpenNMT/...` special case from the tokenizer path entirely.

### Work

- [x] **2.1 Define an adapter interface** in a new `nllb_component/tokenizers.py`. CTranslate2 works
      in **token strings**, not ids, so the interface is string-oriented:

      class NllbTokenizerBackend(Protocol):
          def encode(self, texts: list[str], src_lang: str) -> list[list[str]]: ...
          def decode(self, token_lists: list[list[str]]) -> list[str]: ...
          def count_tokens(self, text: str) -> int: ...

- [x] **2.2 `SentencePieceBackend`** (current behavior, stays the default):
      - encode: `[src_lang] + sp.encode_as_pieces(t) + ["</s>"]`
      - decode: `sp.decode(tokens)`
      - count_tokens: `len(sp.encode_as_pieces(t)) + 2`
- [x] **2.3 `HuggingFaceBackend`**:
      - encode: set `tok.src_lang`, then `tok.convert_ids_to_tokens(tok(t).input_ids)` — already
        includes the `src_lang` prefix and `</s>`, so do **not** add them again
      - decode: `tok.decode(tok.convert_tokens_to_ids(tokens), skip_special_tokens=True)`
      - count_tokens: `len(tok(t).input_ids)`
      - Reload when `src_lang` changes (mirror `develop`'s `_load_tokenizer`, lines 122–135)
- [x] **2.4 Select via job property `NLLB_TOKENIZER`** = `SENTENCEPIECE` (default) | `HUGGINGFACE`.
      Defaulting to SentencePiece preserves exactly today's validated behavior; the property makes
      the alternative a config change rather than a code change.
- [~] **2.5 A/B the backends — small-scale done, full run still owed.** On 20 pt-en
      sentences through the built GPU image, the two backends produced **byte-identical
      translations (20/20)**, and `count_tokens` agreed on **300/300** sentences — so chunk
      boundaries are backend-independent, which is what actually mattered. `encode` diverged on
      2 of 50 sentences, and inspection confirms it is only the known em-dash surface form
      (`'—'` vs `'<unk>'`, same token counts); CTranslate2 maps both to `<unk>`, which is why the
      translations still match. The SentencePiece path is also byte-identical to the
      pre-refactor baseline, so the abstraction is behaviour-preserving.
      Remaining: the full 5,000-sentence scored run (|ΔBLEU| < 0.2, no significant ΔCOMET)
      before *changing the default*. Given 20/20 identical output, that is a formality rather
      than a risk.

- [x] **2.6 Caching.** `_load_tokenizer()` is currently called on **every** `_get_translation()`
      (`nllb_translation_component.py:203`), reloading the 4.8 MB SPM per translation. Load once and
      cache, keyed by backend + `src_lang`.

## Phase 3 — Port `develop`'s token-based splitter (highest quality value)

This is the fix for the −8.6 BLEU Chinese regression. Source: `develop`
`nllb_translation_component.py:178–185` (`_get_text_size_function`) and `197–337` (`_get_translation`).

- [x] **3.1 Wire `count_tokens` into the size function.** Now delegates to the Phase 2 backend,
      so token-based splitting sizes chunks against whichever tokenizer will encode them. Verified
      the two backends agree on 300/300 sentences, so switching backend cannot shift chunking. `develop` uses
      `lambda txt: len(self._tokenizer(txt)["input_ids"])`, which is HF-specific. Replace with
      `self._tokenizer_backend.count_tokens` so token-based splitting works for **both** backends.
      This is the reason the adapter must expose `count_tokens`.
- [x] **3.2 Port the splitter call** — landed via the merge, including `split_mode`,
      `newline_behavior`, and
      `preferred_limit`. **No SDK change needed** — the shared `nlp_text_splitter` already has the
      full signature (`.../nlp_text_splitter/__init__.py:404`).
- [x] **3.3 Change `SENTENCE_MODEL` default** `wtp-bert-mini` → `sat-3l-sm` — done via
      `develop`'s descriptor and `JobConfig`.
- [ ] **3.4 Port the difficult-language logic** (`_is_difficult_language`, `_ARABIC_FLORES_LANGS`,
      `PROCESS_DIFFICULT_LANGUAGES`, `DIFFICULT_LANGUAGE_TOKEN_LIMIT`) — carried over by the merge.
      **Open question: does the 50-token default do anything at all?** Re-running the ar-en
      decomposition with `DIFFICULT_LANGUAGE_TOKEN_LIMIT=0` produced *identical* results, even
      though 13.4% of ar-en sentences exceed 50 tokens — so on per-sentence input the override
      appears to be inert. One untested explanation is that the splitter cannot subdivide a single
      sentence. Worth establishing whether the setting is load-bearing in document mode before
      trusting it; a property that silently does nothing is worse than one tuned wrong.
      *(An earlier version of this plan cited "Arabic scored 38.29 in the decomposition vs 40.71 in
      Axis A" as evidence the limit hurts. That comparison was across different samples — n=1,000 vs
      n=5,000 — and the controlled re-run has since disproved the causal claim.)*
- [ ] **3.4a Do not assume `develop`'s splitter is uniformly better.** As-deployed, `develop`
      under-generates on **Bengali (length ratio 0.692)** and **Persian (0.764)** while the CT2
      branch reaches 0.825 / 0.899 and wins by ~7 BLEU on both. Porting the splitter fixes Chinese
      but may import a bn/fa regression. Gate the port on Axis B for **zh, bn, and fa** — not zh
      alone.
- [ ] **3.5 Consider `SENTENCE_SPLITTER_MODE=SENTENCE` as the CT2 default.** It yields one sentence
      at a time, which pairs naturally with CT2's batch translation — the engine can batch the
      sentences that the splitter emits, instead of translating a few large chunks. This is a
      CT2-specific opportunity `develop` cannot exploit. Validate against `DEFAULT` before adopting.
- [ ] **3.6 Map `max_length` → `max_decoding_length`.** `develop` passes `max_length=hard_limit` to
      `generate()`; the CT2 equivalent on `translate_batch` is `max_decoding_length`. Do not drop it.

## Phase 4 — Fix the model lifecycle bug

`_check_model` (`nllb_translation_component.py:185–188`) only reloads
`if not self._model.model_is_loaded` — never on a name change; the code says
`# TODO: this doesn't do much`. **A job-level `NLLB_MODEL` property is therefore silently ignored**
and the baked-in default is always used. This was discovered the hard way during the evaluation
(it invalidated a whole decomposition run) and is a genuine defect, not just an eval artifact.

- [x] **4.1 Track the loaded model name** — `self._current_model_name` is now set at load time.
      **It is not yet consulted**, so the bug is still live; 4.2 is what actually fixes it.
      (`ctranslate2.Translator` has no
      `name_or_path`, so `develop`'s approach at line 170 does not port directly — store it
      ourselves at load time).
- [ ] **4.2 Reload on mismatch**, resetting the tokenizer backend too.
- [ ] **4.3 Guard against silent fallback.** If a requested `NLLB_MODEL` directory does not exist,
      raise `DetectionException(INVALID_PROPERTY)` rather than loading something else. The current
      `_load_model` "download" branch (line 144-147) calls
      `ctranslate2.Translator(model_path)` on a path that does not exist and then
      `self._model.save_pretrained(...)` — **`Translator` has no `save_pretrained` method**, so that
      branch is dead code that would raise `AttributeError`. Delete it; CT2 models cannot be pulled
      from the Hub as-is anyway.

## Phase 5 — Decode & batching parameters

Currently hardcoded at `nllb_translation_component.py:245–250`: `beam_size = 4`,
`max_batch_size=2024`, `batch_type="tokens"`.

- [~] **5.1 Expose as properties** — `NLLB_BEAM_SIZE` (4) and `NLLB_MAX_BATCH_SIZE` (2024) are in
      `JobConfig` at the values the CT2 branch hardcoded, so behaviour is unchanged; **descriptor
      entries are still owed**. Remaining:
      `NLLB_BATCH_TYPE` (`tokens`). Optionally `length_penalty`, `no_repeat_ngram_size`. For the CPU
      build also consider `inter_threads` / `intra_threads` (see 1.9).
- [ ] **5.2 Keep beam 4. Never inherit `develop`'s greedy default.** As shipped, `develop` decodes
      **greedily** (its `generation_config.json` sets no `num_beams`) while the CT2 path uses beam 4.
      This is not a footnote — it is the most likely explanation for `develop` under-generating on
      Bengali and Persian as-deployed, where CT2 wins by **~7 BLEU** with length ratios of 0.825/0.899
      against `develop`'s 0.692/0.764. Axis A rules out the model (bn −0.10, fa +0.02 with decoding
      harmonized), so the gap comes from the pipeline. Corollary for reporting: moving to CT2 improves
      quality vs today's fp16 deployment partly *because of beam search*, so do not credit that gain
      to CTranslate2 or to precision.
- [ ] **5.2a Separate splitter from decoding before drawing splitter conclusions.** Axis B confounds
      the two. The cheap discriminating experiment: re-run the `develop` as-deployed blob for bn/fa
      with beam 4 forced (the eval driver already does this by setting
      `generation_config.num_beams = 4` at runtime — no rebuild needed). If the bn/fa gap closes, it
      was decoding; if it persists, `develop`'s splitter genuinely mishandles those scripts and
      task 3.4a becomes a blocker rather than a check.
- [x] **5.3 Clean up the `should_translate` batch hack** — done in the merge: chunks are filtered
      *before* batching and re-inserted by index, rather than translated and discarded. Previously —
      including ones that should not be translated — is sent to the model, and the output is then
      discarded and swapped back. Instead, filter before the call and re-insert by index. Saves GPU
      work and removes the "Temporary hack" comment.

## Phase 6 — Descriptor & property reconciliation ✅ (via the merge)

All resolved by taking `develop`'s descriptor in `8d300300`.

| Property | develop | CT2 | Outcome |
|---|---|---|---|
| `USE_NLLB_TOKEN_LENGTH` | ✅ | ❌ | added |
| `NLLB_TRANSLATION_TOKEN_LIMIT` | ✅ 512 | ❌ | added |
| `NLLB_TRANSLATION_TOKEN_SOFT_LIMIT` | ✅ 130 | ❌ | added |
| `SENTENCE_SPLITTER_MODE` | ✅ | ❌ | added |
| `SENTENCE_SPLITTER_NEWLINE_BEHAVIOR` | ✅ | ❌ | added |
| `PROCESS_DIFFICULT_LANGUAGES` | ✅ | ❌ | added |
| `DIFFICULT_LANGUAGE_TOKEN_LIMIT` | ✅ | ❌ | added |
| `IS_ANNOTATOR` | ✅ | ❌ | added |
| ~~`OUTPUT_MERGE_WITH_PREVIOUS_TASK`~~ | ❌ | ✅ | **dropped — see 6.3** |
| `SENTENCE_MODEL` | `sat-3l-sm` | `wtp-bert-mini` | default now `sat-3l-sm` |
| `NLLB_MODEL` default | `facebook/...` | `OpenNMT/...ct2-int8` | CT2 model kept |
| `NLLB_TOKENIZER` | — | — | still **new** (Phase 2) |
| `NLLB_BEAM_SIZE`, `NLLB_MAX_BATCH_SIZE` | — | — | in `JobConfig`; descriptor entries still owed (5.1) |

Also inherited from `develop`: the actions/tasks/pipelines were renamed from
"NO LANGUAGE LEFT BEHIND TRANSLATION …" to "NLLB TRANSLATION …". Anything referencing the old
names by string needs updating.

- [x] **6.1 Merge `nllb_utils.py` — casing hazard.** Resolved as planned: `develop`'s table and
      `.lower()` normalisation, with CT2's `_flores_to_wtpsplit_iso_639_1` and `get_normalized_iso`
      spliced on top. Verified `get_code("POR","LATN") == "por_Latn"`.
- [x] **6.2 Keep CT2's `get_normalized_iso`** — retained.
- [x] **6.3 `OUTPUT_MERGE_WITH_PREVIOUS_TASK` is retired, not missing.** *This corrects an error in
      the earlier version of this plan, which listed it as "keep" based on a property-name diff
      alone.* Commit `cb351e4a` renamed it to `IS_ANNOTATOR` **repo-wide** (the diff for
      AzureTranslation is literally `-OUTPUT_MERGE_WITH_PREVIOUS_TASK` / `+IS_ANNOTATOR`), and it
      appears in **zero** descriptors on `develop`. Re-adding it would have resurrected a dead
      property. Dropped.
- [ ] **6.4 Known gap: `zho_Hans` is absent from `_flores_to_wtpsplit_iso_639_1`.** Pre-existing on
      the CT2 branch, not introduced by the merge — `get_normalized_iso("zho_Hans")` returns the
      input unchanged, so the WtP/SaT adaptor language falls back rather than resolving to `zh`
      (`yue_Hant` does map). Worth fixing given Chinese is the worst-performing pair.

## Phase 7 — Tests

> **Blocking.** The suite is currently **red** and `RUN_TESTS=true` will fail the build. Measured
> against the merge result in the ctranslate2 image: **25 tests — 9 pass, 15 fail, 1 errors.**
> None indicate a functional defect. `develop`'s suite was taken wholesale (it is a strict superset
> of the CT2 one by test name), so its expectations encode `develop`'s model and decoding.

- [ ] **7.1 Re-baseline the 15 expected-output failures.** Every one asserts a translation string
      produced by **facebook fp16 with greedy decoding**; this component runs **CT2 int8 at beam 4**,
      which legitimately words things differently — `'Hello, how are you today?'` vs `'Hi, …'`,
      `'It is raining.'` vs `"It's raining."`, and longer paraphrase differences on the
      paragraph/wtp tests. Decide whether the canonical expectations should track the shipped
      configuration (regenerate) or be loosened to assertions that are not decoder-specific.
      Regenerating pins the tests to whatever `BUILD_TYPE` produced them, which matters now that
      gpu→fp16 and cpu→int8 give different output.
- [ ] **7.1a Fix the one erroring test.**
      `test_difficult_language_token_limit_overrides_soft_limit_not_hard_limit` calls
      `component._tokenizer(text)["input_ids"]` — the HuggingFace convention — on a
      `SentencePieceProcessor`, raising `TypeError: 'SentencePieceProcessor' object is not callable`.
      It is white-box against the tokenizer. Either route it through the Phase 2 adapter's
      `count_tokens` or use `len(component._tokenizer.encode_as_pieces(text)) + 2`.
      *Currently passing (9), so these are the regression guard in the meantime:*
      `test_invalid_script_lang_combination`, `test_long_spanish`, `test_sentence_split_job`,
      `test_should_translate`, `test_unsupported_{source,target}_{language,script}`,
      `test_wtp_iso_conversion`.
- [ ] **7.2 Tokenizer parity test.** Assert both backends produce CT2-compatible token strings and
      round-trip a fixed corpus. Encode the two known divergences (trailing whitespace, unknown-char
      surface form) as *expected*, so they do not read as regressions.
- [ ] **7.3 Model-swap test** for Phase 4 — assert `NLLB_MODEL` actually changes the loaded model,
      and that a bogus name raises rather than silently falling back.
- [ ] **7.4 Keep `RUN_TESTS` build arg working** for **both** `BUILD_TYPE` values, since each now
      produces a different model artifact. The CPU build is the one likely to time out.

## Phase 8 — Validation

- [ ] **8.1 Re-run Axis A** (`eval/run_pipeline.sh`) on the new GPU image for at least pt/ar/zh plus
      one Cyrillic and one Indic pair. **Baseline: the fp16 column of `eval/pipeline-results/`**, not
      the int8 column — the GPU build now ships fp16. Axis A proved hardware-independent (H100
      reproduced the RTX 5070 Ti scores to 3 decimals), so the recorded numbers are a valid target.
      Acceptance: no significant regression. Any large delta points at a decode-parameter or
      tokenizer change, not at the model, since Δquant is non-significant on all 9 pairs.
- [ ] **8.2 Re-run Axis B** (`RUN_AXIS_B=1`) for **zh, bn, and fa** — the acceptance test for
      Phase 3. Acceptance: **zh length ratio reaches ~0.90**, in line with the healthy languages.
      Note the earlier target of "recover toward `develop`'s 0.751" was **wrong**: 0.751 is itself
      deficient, and `develop` is worse still on bn (0.692) and fa (0.764). Do not regress bn/fa
      below the CT2 branch's current 0.825 / 0.899.
- [ ] **8.3 Re-measure throughput on the target hardware.** Expect ~2.3× over HF-fp16 on H100, not
      the ~6× seen on a consumer card. Record CPU-build throughput separately — it sets whether the
      CPU target is viable for the intended workload at all.
- [ ] **8.4 Smoke-test the CPU build end-to-end** (`--build-arg BUILD_TYPE=cpu`). Confirm the
      converted model loads with `compute_type=int8_float32`, that `_resolve_device()` correctly
      falls back to CPU, and that a short job completes. This path is now a shipped configuration,
      not an option — it needs its own gate.
- [x] **8.5 Decomposition Arabic parity — harness updated, but the confound was not real.**
      `run_decomp.sh` now passes `DIFFICULT_LANGUAGE_TOKEN_LIMIT=0` to `hf-fp16`, matching
      `run_pipeline.sh`. **The hypothesis this was meant to fix has been disproved:** re-running
      ar-en with the property set produced *identical* results, so the +1.91 BLEU / +0.71 COMET
      engine effect is **not** a chunking artifact. The change is harmless parity insurance, not a
      correction.
- [x] **8.5a Re-run the ar-en decomposition — done, results unchanged.** The committed data stands
      as valid. ar-en remains an unexplained outlier: the only pair with a large Δengine, and the
      slowest HF-fp16 throughput (1.23 vs ~2.2 sent/s). Reported as measured in `REPORT.md` rather
      than excluded.
- [ ] **8.5b Optional: establish *why* ar-en is an outlier.** Two explanations are dead (chunking
      confound; sentences too short). A repeat on a second Arabic-script corpus would show whether
      the effect is language-specific or sample-specific. Not blocking — it changes no decision in
      this plan, since the engine choice does not hinge on one pair.

## Phase 9 — Before the merge request

- [ ] **9.1 Delete `python/NllbTranslation/eval/`.** It is a prototype enabler, not product code
      (see the banner at the top of this document). `git rm -r python/NllbTranslation/eval` — nothing
      under `nllb_component/`, `plugin-files/`, `tests/` or the `Dockerfile` imports or references
      it, so removal is clean. The evaluation record is preserved on the
      `eval/nllb-mt-evaluation` branch.
- [ ] **9.2 Confirm nothing else prototype-only leaks into the MR** — e.g. `Dockerfile.dev`, any
      scratch models under `models/`, and the `TODO (Phase N)` comments seeded in
      `nllb_translation_component.py` during the merge.
- [ ] **9.3 Decide the fate of this document.** `PLAN.md` is itself prototype scaffolding; either
      drop it from the MR or reduce it to whatever design notes are worth keeping in-tree.

---

## Risks

| Risk | Mitigation |
|---|---|
| Every build now pulls 17 GB and converts | Accepted cost of single-source provenance; multi-stage keeps it out of the image; confirm CI disk before merge |
| Wrong precision for the target device (fp16 on CPU silently becomes float32; `int8_float16` on CPU hard-errors) | `BUILD_TYPE` drives conversion (1.1); `compute_type` asserted and logged at load (1.5) |
| Bare `int8` means `int8_float32` — a trap on GPU, correct on CPU | Documented in 1.6 so nobody "fixes" the CPU build to `int8_float16` |
| Porting `develop`'s splitter imports its bn/fa under-generation | Gate on Axis B for zh **and** bn/fa (3.4a, 8.2); keep beam 4 (5.2) |
| CPU target may be too slow to be usable | Measure in 8.3/8.4 before promising it |
| `develop`↔CT2 merge conflicts across all four files | Phase 0 first, in its own commit, before any feature work |
| `nllb_utils` casing mismatch silently breaks language lookup | Explicit task 6.1 + test over a language/script matrix |
| Tokenizer swap changes output subtly | Default stays SentencePiece; A/B gate (2.5) before any default change |

## Open questions

1. ~~fp16 or int8? Convert ourselves or use the prebuilt?~~ **Decided: both precisions, one source,
   selected by `BUILD_TYPE` — `gpu` → fp16, `cpu` → int8, both converted from
   `facebook/nllb-200-3.3B`.** Quality is not a factor (Δquant non-significant on all 9 pairs), so
   the choice is purely device fit: on H100 fp16 is faster than int8 on every pair, and on CPU fp16
   is not even supported. Dropping the `OpenNMT` prebuilt costs a 17 GB conversion on every build
   and buys a single provenance with one pinned revision — plus, since its SPM is byte-identical to
   the source checkpoint's, it costs nothing in tokenizer fidelity.
2. **Should `SENTENCE` split mode become the CT2 default** (task 3.5)? It suits batch translation
   but changes chunking behavior for every job.
3. **Ship the HF tokenizer backend enabled or behind a flag?** It pulls `transformers` into the
   runtime dependency set — already present in the image today, but making it a declared dependency
   is a supply-chain decision.
4. ~~`OUTPUT_MERGE_WITH_PREVIOUS_TASK` on the CT2 branch only — intentional?~~ **Answered: it was
   stale.** `develop` renamed it to `IS_ANNOTATOR` repo-wide in `cb351e4a`. Dropped in the merge —
   see task 6.3.
5. **Which `BUILD_TYPE` should the test expectations be baselined against** (task 7.1)? **Now
   demonstrated, not hypothetical:** on a 10-sentence sample the gpu (`float16`) and cpu
   (`int8_float32`) images agreed on 8 and diverged on 2. A regenerated expected-string suite would
   therefore pass on one build target and fail on the other. Options: baseline on gpu and skip the
   strings on cpu; assert decoder-agnostically (length ratio, non-empty, no source passthrough) and
   test less; or keep two expectation sets. This blocks 7.1.

> **Note on eval-harness paths.** Tasks referencing `eval/…` scripts now resolve on this branch —
> the harness was copied here as a prototype enabler (see the banner at the top). The *results*
> (`REPORT.md`, `pipeline-results/`) deliberately were not, and still live on
> **`eval/nllb-mt-evaluation`**.
