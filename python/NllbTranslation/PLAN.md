# NllbTranslation: CTranslate2 — Implementation Plan

**Target branch:** `regexer/nllb-ctranslate2-updates` (base), drawing from `develop`.
**Goal:** ship the NllbTranslation component on the **CTranslate2** inference engine with
`facebook/nllb-200-3.3B` converted to **`int8_float16`** (with **fp16 available as a one-flag build
option**), while keeping the **tokenizer backend swappable** (FLORES SentencePiece today,
HuggingFace `AutoTokenizer` as a first-class alternative).

**Evidence base:** `eval/REPORT.md`. The three findings that drive this plan:

- The ~6× throughput win is the **engine**, not the quantization. CT2-fp16 = 6.3× HF-fp16 and
  CT2-int8 = 6.6×, all at statistically indistinguishable quality.
- **Quantization is a no-op for quality.** Holding the engine fixed, int8-vs-fp16 is within noise on
  every metric (ΔBLEU −0.04 p=0.83, ΔchrF +0.04 p=0.80, ΔCOMET −0.00 p=0.96). This is what makes
  `int8_float16` the right default: it is quality-equivalent, marginally faster, and half the disk.
- The as-deployed int8 quality gap (−8.6 BLEU on Chinese) is the **character-based splitter**, not
  the model. Porting `develop`'s token-based splitter is the single highest-value code change here.

**Branch topology.** `regexer/nllb-ctranslate2-updates` forked from `develop` at `9adca039`
(Feat/py3.12). `develop` has since landed `b22c3f55` (TextSplitter utility update — the token-based
splitter), `cb351e4a` (IS_ANNOTATOR / SUPPRESS_TRACKS), and `7e9b4e9d` (version 10.0). None are on
the CT2 branch.

---

## Phase 0 — Branch hygiene

- [ ] **0.1 Sync with `develop`.** Merge or rebase `develop` into `regexer/nllb-ctranslate2-updates`.
      Expect conflicts in all four main files. Resolution policy: take `develop` for splitter/config
      logic, keep CT2 for the inference call path.
- [ ] **0.2 Fix `setup.cfg`.** Bump `version` 9.0 → 10.0 and `mpf_component_{api,util}>=9.0` →
      `>=10.0`. **Remove the bogus self-dependency `NllbTranslation` from `install_requires`.**
      Keep `ctranslate2>=4.5.0`, `sentencepiece>=0.2.0`; add `transformers` explicitly if the HF
      tokenizer backend ships enabled (it is currently present transitively — the CT2 image already
      has transformers 5.13.0).
- [ ] **0.3 Descriptor version.** `componentVersion` / `middlewareVersion` 9.0 → 10.0. Note
      `develop` has 2 pipelines vs CT2's 1 — reconcile.

## Phase 1 — Model conversion & packaging (`int8_float16` default, fp16 optional)

We convert from `facebook/nllb-200-3.3B` at build time rather than pulling the prebuilt
`OpenNMT/nllb-200-3.3B-ct2-int8`. This is a **provenance change, not a numerics change**: the
OpenNMT model's stored compute type already resolves to `int8_float16`, so converting ourselves at
`int8_float16` reproduces today's deployed arithmetic while giving us a pinned upstream revision, a
first-party artifact, and control over which tokenizer files ship in the model dir (see 1.2).
It also means fp16 is reachable by changing one flag — there is no published fp16 CTranslate2 build
of NLLB-200-3.3B, so the fp16 option *requires* build-time conversion regardless.
Validated recipe is in `eval/convert_ct2.sh`.

- [ ] **1.1 Add a `convert_model` build stage** to `Dockerfile`, replacing the current
      `download_model` stage that pulls `OpenNMT/nllb-200-3.3B-ct2-int8`:

      ARG MODEL_NAME=facebook/nllb-200-3.3B
      ARG MODEL_REVISION=1a07f7d195896b2114afcb79b7b57ab512e7b43e
      ARG CT2_QUANTIZATION=int8_float16
      RUN pip install -U "huggingface_hub[cli]" ctranslate2 transformers sentencepiece
      RUN ct2-transformers-converter --model $MODEL_NAME --revision $MODEL_REVISION \
            --output_dir /models/nllb-200-3.3B-ct2-$CT2_QUANTIZATION \
            --quantization $CT2_QUANTIZATION --low_cpu_mem_usage \
            --copy_files tokenizer.json tokenizer_config.json special_tokens_map.json \
                         sentencepiece.bpe.model

- [ ] **1.2 `--copy_files` is mandatory, not optional.** Verified: a bare
      `ct2-transformers-converter` run emits only `config.json`, `model.bin`,
      `shared_vocabulary.json` — **no tokenizer files**. Without `--copy_files` the HuggingFace
      tokenizer backend (Phase 2) is impossible and the model dir is not self-describing. The
      production `OpenNMT/nllb-200-3.3B-ct2-int8` dir *does* ship `tokenizer.json`,
      `tokenizer_config.json`, `special_tokens_map.json` — match that.
- [ ] **1.3 Keep the FLORES SPM download** (`OpenNMT/nllb-200-onmt/flores200_sacrebleu_tokenizer_spm.model`)
      so the SentencePiece backend stays available. It is 4.8 MB; cost of keeping both is trivial.
- [ ] **1.4 Make quantization a build ARG.** `CT2_QUANTIZATION=int8_float16` default; fp16 is a
      drop-in via `--build-arg CT2_QUANTIZATION=float16`.
      **Never use bare `int8`** — it produces a model whose stored default resolves to
      `int8_float32`: essentially lossless, but with *no* tensor-core speedup (measured identical
      sent/s to fp16, and 0/1000 outputs changed). `int8_float16` is the fast, deployed
      configuration. This bit us mid-evaluation; see `eval/REPORT.md`.
- [ ] **1.5 Assert the resolved compute type at load.** Because the bare-`int8` trap is silent, log
      `Translator.compute_type` after loading and fail loudly if it is not the requested one. The
      standalone `eval/ct2_driver.py` already records `actual_compute_type` for exactly this reason
      — reuse the check.
- [ ] **1.6 Update `DEFAULT_NLLB_MODEL`** in `nllb_translation_component.py:49` from
      `'OpenNMT/nllb-200-3.3B-ct2-int8'` to the converted dir name (include the quantization in the
      name so the deployed variant is self-evident, e.g. `nllb-200-3.3B-ct2-int8_float16`).
- [ ] **1.7 Budget the size.** `int8_float16` = **3.36 GB** on disk; the fp16 option is **6.7 GB**.
      Either way the build stage transiently needs the **17 GB** HF checkpoint plus the conversion
      output. Confirm CI builder disk headroom before merging — a WSL2 disk exhaustion already bit
      this project once. Multi-stage build means only the converted dir lands in the final image.

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

### Work

- [ ] **2.1 Define an adapter interface** in a new `nllb_component/tokenizers.py`. CTranslate2 works
      in **token strings**, not ids, so the interface is string-oriented:

      class NllbTokenizerBackend(Protocol):
          def encode(self, texts: list[str], src_lang: str) -> list[list[str]]: ...
          def decode(self, token_lists: list[list[str]]) -> list[str]: ...
          def count_tokens(self, text: str) -> int: ...

- [ ] **2.2 `SentencePieceBackend`** (current behavior, stays the default):
      - encode: `[src_lang] + sp.encode_as_pieces(t) + ["</s>"]`
      - decode: `sp.decode(tokens)`
      - count_tokens: `len(sp.encode_as_pieces(t)) + 2`
- [ ] **2.3 `HuggingFaceBackend`**:
      - encode: set `tok.src_lang`, then `tok.convert_ids_to_tokens(tok(t).input_ids)` — already
        includes the `src_lang` prefix and `</s>`, so do **not** add them again
      - decode: `tok.decode(tok.convert_tokens_to_ids(tokens), skip_special_tokens=True)`
      - count_tokens: `len(tok(t).input_ids)`
      - Reload when `src_lang` changes (mirror `develop`'s `_load_tokenizer`, lines 122–135)
- [ ] **2.4 Select via job property `NLLB_TOKENIZER`** = `SENTENCEPIECE` (default) | `HUGGINGFACE`.
      Defaulting to SentencePiece preserves exactly today's validated behavior; the property makes
      the alternative a config change rather than a code change.
- [ ] **2.5 A/B the backends** using the existing harness: run `eval/nllb_eval_driver.py`
      per-sentence over an existing 5,000-sentence sample with each backend and score with
      `eval/mt_eval.py compare`. Acceptance: |ΔBLEU| < 0.2 and no significant ΔCOMET. Only then
      consider changing the default.
- [ ] **2.6 Caching.** `_load_tokenizer()` is currently called on **every** `_get_translation()`
      (`nllb_translation_component.py:203`), reloading the 4.8 MB SPM per translation. Load once and
      cache, keyed by backend + `src_lang`.

## Phase 3 — Port `develop`'s token-based splitter (highest quality value)

This is the fix for the −8.6 BLEU Chinese regression. Source: `develop`
`nllb_translation_component.py:178–185` (`_get_text_size_function`) and `197–337` (`_get_translation`).

- [ ] **3.1 Wire `count_tokens` into the size function.** `develop` uses
      `lambda txt: len(self._tokenizer(txt)["input_ids"])`, which is HF-specific. Replace with
      `self._tokenizer_backend.count_tokens` so token-based splitting works for **both** backends.
      This is the reason the adapter must expose `count_tokens`.
- [ ] **3.2 Port the splitter call** including `split_mode`, `newline_behavior`, and
      `preferred_limit`. **No SDK change needed** — the shared `nlp_text_splitter` already has the
      full signature (`.../nlp_text_splitter/__init__.py:404`).
- [ ] **3.3 Change `SENTENCE_MODEL` default** `wtp-bert-mini` → `sat-3l-sm`.
- [ ] **3.4 Port the difficult-language logic** (`_is_difficult_language`, `_ARABIC_FLORES_LANGS`,
      `PROCESS_DIFFICULT_LANGUAGES`, `DIFFICULT_LANGUAGE_TOKEN_LIMIT`).
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

- [ ] **4.1 Track the loaded model name** in an instance attribute (`ctranslate2.Translator` has no
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

- [ ] **5.1 Expose as properties:** `NLLB_BEAM_SIZE` (default 4), `NLLB_MAX_BATCH_SIZE` (2024),
      `NLLB_BATCH_TYPE` (`tokens`). Optionally `length_penalty`, `no_repeat_ngram_size`.
- [ ] **5.2 Note the expected quality change vs `develop`.** As shipped, `develop` decodes
      **greedily** (its `generation_config.json` sets no `num_beams`), while the CT2 path uses beam 4.
      Moving to CT2 should therefore *improve* quality relative to today's fp16 deployment,
      independent of engine or precision. Do not attribute that gain to CTranslate2.
- [ ] **5.3 Clean up the `should_translate` batch hack** (lines 258–266). Today every segment —
      including ones that should not be translated — is sent to the model, and the output is then
      discarded and swapped back. Instead, filter before the call and re-insert by index. Saves GPU
      work and removes the "Temporary hack" comment.

## Phase 6 — Descriptor & property reconciliation

CT2 branch is missing 8 properties present on `develop`, and has 1 `develop` lacks.

| Property | develop | CT2 | Action |
|---|---|---|---|
| `USE_NLLB_TOKEN_LENGTH` | ✅ | ❌ | add (Phase 3) |
| `NLLB_TRANSLATION_TOKEN_LIMIT` | ✅ 512 | ❌ | add |
| `NLLB_TRANSLATION_TOKEN_SOFT_LIMIT` | ✅ 130 | ❌ | add |
| `SENTENCE_SPLITTER_MODE` | ✅ | ❌ | add |
| `SENTENCE_SPLITTER_NEWLINE_BEHAVIOR` | ✅ | ❌ | add |
| `PROCESS_DIFFICULT_LANGUAGES` | ✅ | ❌ | add |
| `DIFFICULT_LANGUAGE_TOKEN_LIMIT` | ✅ | ❌ | add |
| `IS_ANNOTATOR` | ✅ | ❌ | add (from `cb351e4a`) |
| `OUTPUT_MERGE_WITH_PREVIOUS_TASK` | ❌ | ✅ | keep |
| `SENTENCE_MODEL` | `sat-3l-sm` | `wtp-bert-mini` | change default |
| `NLLB_TOKENIZER` | — | — | **new** (Phase 2) |
| `NLLB_BEAM_SIZE`, `NLLB_MAX_BATCH_SIZE`, `NLLB_BATCH_TYPE` | — | — | **new** (Phase 5) |

- [ ] **6.1 Merge `nllb_utils.py` carefully — casing hazard.** The two branches' language tables use
      **different key casing**: `develop` uses lowercase script keys (`'latn'`) with `.lower()`
      normalization in `get_code`; CT2 uses capitalized keys (`'Latn'`) with no normalization.
      Dropping `develop`'s `get_code` onto CT2's table silently breaks every lookup. Adopt
      `develop`'s table + normalization + `DetectionException` messaging, then port CT2's additions
      (`_flores_to_wtpsplit_iso_639_1`, `get_normalized_iso`) on top. CT2's version also lost the
      `mpf` import and raises a bare `KeyError`; `develop`'s raises a proper
      `DetectionException` naming the bad language/script pair.
- [ ] **6.2 Keep CT2's `get_normalized_iso` in the splitter path.** CT2 calls
      `WtpLanguageSettings.convert_to_iso(NllbLanguageMapper.get_normalized_iso(lang))`, which
      handles FLORES→ISO-639-1 codes `develop` misses. Combine with `develop`'s fallback chain
      (lines 256–264).

## Phase 7 — Tests

- [ ] **7.1 Reconcile the test suites** (CT2 852 lines vs `develop` 1054). Port `develop`'s splitter
      and token-limit tests.
- [ ] **7.2 Tokenizer parity test.** Assert both backends produce CT2-compatible token strings and
      round-trip a fixed corpus. Encode the two known divergences (trailing whitespace, unknown-char
      surface form) as *expected*, so they do not read as regressions.
- [ ] **7.3 Model-swap test** for Phase 4 — assert `NLLB_MODEL` actually changes the loaded model,
      and that a bogus name raises rather than silently falling back.
- [ ] **7.4 Keep `RUN_TESTS` build arg working** with the converted model (and with the fp16 option,
      which is 2× the size).

## Phase 8 — Validation

- [ ] **8.1 Re-run Axis A** (`eval/run_pipeline.sh`) against the new CT2 image for at least
      pt/ar/zh plus one Cyrillic and one Indic pair. Acceptance: no significant regression vs the
      recorded int8 numbers in `eval/REPORT.md`. Our own `int8_float16` conversion should be
      *numerically comparable* to the OpenNMT prebuilt it replaces — a large delta here means the
      conversion is wrong, not that the model changed.
- [ ] **8.2 Re-run Axis B** (`RUN_AXIS_B=1`) for **zh-en specifically** — this is the acceptance
      test for Phase 3. Acceptance: length ratio recovers from **0.570** toward `develop`'s
      **0.751**, and ΔBLEU vs fp16-develop is no longer ≈ −8.6.
- [ ] **8.3 Re-measure throughput** to confirm the ~6× holds in the packaged image.
- [ ] **8.4 Spot-check the fp16 option** (`--build-arg CT2_QUANTIZATION=float16`) on one pair, so
      the fallback is known-good rather than theoretical. Record whether fp16's larger footprint
      changes batching headroom on a 16 GB card.

---

## Risks

| Risk | Mitigation |
|---|---|
| Build-time conversion adds ~17 GB transient disk + a long build step | Multi-stage; pin revision; confirm CI disk before merge |
| Bare `int8` silently yields the slow `int8_float32` path | Pin `int8_float16` in the ARG + assert `compute_type` at load (1.5) |
| Self-converted model diverges from the OpenNMT prebuilt it replaces | Axis A re-run (8.1) is the regression gate; same compute type either way |
| fp16 option is 2× the disk (6.7 vs 3.36 GB) | Not the default; `CT2_QUANTIZATION` ARG keeps it one flag away, spot-checked in 8.4 |
| `develop`↔CT2 merge conflicts across all four files | Phase 0 first, in its own commit, before any feature work |
| `nllb_utils` casing mismatch silently breaks language lookup | Explicit task 6.1 + test over a language/script matrix |
| Tokenizer swap changes output subtly | Default stays SentencePiece; A/B gate (2.5) before any default change |

## Open questions

1. ~~Is fp16 worth 2× the disk over `int8_float16`?~~ **Decided: `int8_float16`.** The evaluation
   shows quantization costs nothing in quality (p≥0.80 on every metric) while halving the model
   footprint, so fp16's only argument — having no quantization to defend — does not justify the
   size. fp16 stays reachable via `--build-arg CT2_QUANTIZATION=float16` and is spot-checked in 8.4.
   *Residual question:* keep converting from `facebook/nllb-200-3.3B` ourselves, or go back to the
   `OpenNMT` prebuilt for int8? Self-conversion buys provenance, a pinned revision, and controlled
   tokenizer files, at the cost of a long build stage.
2. **Should `SENTENCE` split mode become the CT2 default** (task 3.5)? It suits batch translation
   but changes chunking behavior for every job.
3. **Ship the HF tokenizer backend enabled or behind a flag?** It pulls `transformers` into the
   runtime dependency set — already present in the image today, but making it a declared dependency
   is a supply-chain decision.
4. **`OUTPUT_MERGE_WITH_PREVIOUS_TASK` on the CT2 branch only** — intentional, or should it go to
   `develop` too?
