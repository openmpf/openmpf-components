# NLLB MT Quality Eval — portable deployment

Evaluate NLLB-200-3.3B translation quality (fp16 vs int8, or any two OpenMPF
NllbTranslation image variants) on TMX corpora. Produces per-sentence BLEU /
chrF / chrF++ / TER + COMET with paired-bootstrap significance, plus an
"as-deployed" document-level score, and a combined summary across language
pairs.

Two axes per pair:
- **Axis A — intrinsic:** each sentence fed on its own, beam 4 on both models,
  sentence-splitting neutralized. Isolates the model/quantization variable.
- **Axis B — as-deployed:** the whole file through each image's own shipped
  pipeline (its own splitter + decoding). Reflects real production behavior.

---

## Prerequisites (an OpenMPF dev environment already has all of these)

1. **Docker + NVIDIA container runtime + a CUDA GPU.** fp16 needs ~16 GB VRAM at
   batch 16; lower `FP16_BATCH` for smaller cards. int8 needs ~4–6 GB.
2. **The two OpenMPF NllbTranslation images** (this is the only non-pip
   dependency):
   - `openmpf_nllb_translation:develop`  — facebook/nllb-200-3.3B (fp16, PyTorch)
   - `openmpf_nllb_translation:ctranslate2` — OpenNMT/nllb-200-3.3B-ct2-int8 (int8)

   Build them from the `openmpf-components` repo (models download from HF at
   build time — internet required):
   ```bash
   # fp16 (facebook) — from the develop branch
   cd openmpf-components/python/NllbTranslation && git checkout develop
   #   build via your usual OpenMPF component build, tagging it :develop
   # int8 (ctranslate2) — from the working ctranslate2 branch
   git checkout <ctranslate2 branch>   # e.g. feature/nllb-ctranslate2
   #   build, tagging it :ctranslate2
   ```
   If your enclave tags them differently, point the pipeline at your tags:
   ```bash
   FP16_IMAGE=my/nllb:fp16 INT8_IMAGE=my/nllb:int8 ./run_pipeline.sh
   ```
   (You can compare any two image variants, not just fp16 vs int8 — whichever
   two you set as FP16_IMAGE / INT8_IMAGE.)
3. **Python 3 + internet** for the scoring venv (`./setup_venv.sh`).

## Setup

```bash
./setup_venv.sh            # pip-installs sacrebleu, COMET (+torch), etc.
mkdir -p tmx && cp /path/to/*.tmx tmx/     # drop your TMX corpora here
# edit run_pipeline.sh -> PAIRS (see below), then:
./preflight.sh             # verifies docker, GPU, images, venv, tmx files
```

## Configure your language pairs

Edit the `PAIRS` array in `run_pipeline.sh`. Each line is:

```
name | tmx_path | tmx_src_lang | nllb_src_lang | nllb_src_script
```

- `tmx_src_lang` — the `xml:lang` of the **source** side inside the TMX
  (check with: `grep -m4 xml:lang tmx/yourfile.tmx`). Everything is translated
  **into English**, so the reference side is the `en` segments.
- `nllb_src_lang` / `nllb_src_script` — the NLLB / FLORES-200 code, split into
  its language and script halves.

### NLLB / FLORES-200 source codes (common)

| Language | code | lang | script |
|---|---|---|---|
| Arabic (MSA) | arb_Arab | arb | Arab |
| Chinese, Mandarin (Simplified) | zho_Hans | zho | Hans |
| Chinese, Mandarin (Traditional) | zho_Hant | zho | Hant |
| Chinese, Cantonese (Traditional) | yue_Hant | yue | Hant |
| Portuguese | por_Latn | por | Latn |
| Spanish | spa_Latn | spa | Latn |
| French | fra_Latn | fra | Latn |
| German | deu_Latn | deu | Latn |
| Italian | ita_Latn | ita | Latn |
| Russian | rus_Cyrl | rus | Cyrl |
| Japanese | jpn_Jpan | jpn | Jpan |
| Korean | kor_Hang | kor | Hang |
| Hindi | hin_Deva | hin | Deva |

**Watch the variant:** an OPUS "zh" file may be Mandarin *or* Cantonese, and
Simplified *or* Traditional — they need different codes. Inspect the text if
unsure. To list every code the model supports, or to check one:
```bash
docker run --rm --entrypoint /opt/mpf/plugin-venv/bin/python openmpf_nllb_translation:develop \
  -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('/models/facebook/nllb-200-3.3B'); print(t.convert_tokens_to_ids('zho_Hans'))"
# a real id (not the unk id) means the code is valid
```
The pipeline also runs a 3-line smoke per pair and **aborts that pair early if
the code is rejected**, so a bad code won't waste a whole overnight run.

## Run

```bash
./run_pipeline.sh                 # all pairs, both axes (long — run overnight)
./run_pipeline.sh ar-en           # a single pair
RUN_AXIS_B=0 ./run_pipeline.sh    # Axis A only (skips the slow as-deployed blobs)
```

Knobs (env vars): `N` (sentences/pair, default 5000), `SEED`, `GPU`
(`'"device=1"'`), `FP16_IMAGE` / `INT8_IMAGE`, `FP16_BATCH`, `BOOTSTRAP`,
`RUN_AXIS_B`. It is **resumable** — re-running skips finished stages and
generation resumes from where it stopped.

Rough timing (RTX-class GPU, 5000 sentences): int8 ~20–25 min/pair;
fp16 Axis A ~1.5–2 h/pair; fp16 Axis B blob ~2–3 h/pair (the long pole —
set `RUN_AXIS_B=0` to skip).

## Outputs

- `results/SUMMARY.md` — combined fp16-vs-int8 table across all pairs.
- `results/<pair>/`
  - `hyp.fp16.en`, `hyp.int8.en` — aligned per-sentence hypotheses
  - `axisA.report.txt` — metrics table + bootstrap significance
  - `axisA.segments.csv` — per-sentence scores incl. COMET (for error analysis)
  - `asdeployed.{fp16,int8}.json`, `axisB.*.report.txt` — as-deployed axis
  - `meta.*.json` — run metadata/timing
  - `sample.{src,ref,idx}` — the exact sampled test set (reproducible via SEED)

## Files in this folder

| File | Role |
|---|---|
| `run_pipeline.sh` | orchestrator — edit `PAIRS`, then run |
| `tmx_sample.py` | extract + fixed-seed sample from a TMX |
| `nllb_eval_driver.py` | runs **inside** an image; 1 translation per input line (Axis A) |
| `mt_eval.py` | scoring: `compare` (2 systems + COMET + bootstrap) / `score` (1 system) |
| `summarize.py` | builds `results/SUMMARY.md` across pairs |
| `make_sample.py` | optional: sample from moses parallel files instead of TMX |
| `setup_venv.sh` / `preflight.sh` / `requirements.txt` | env setup + readiness check |

## Notes / gotchas

- **fp16 OOM:** batched fp16 fragments GPU memory on long runs. The pipeline
  already sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`, `--batch 16`,
  and periodic cache clearing. Do **not** use `expandable_segments:True` under
  WSL2 (its virtual-memory APIs fail there). Lower `FP16_BATCH` on smaller GPUs.
- **Arabic** is NLLB's flagged "difficult" language; the pipeline passes
  `DIFFICULT_LANGUAGE_TOKEN_LIMIT=0` on the fp16 Axis A run so both images
  segment it identically (keeps Axis A a fair comparison).
- **Cross-engine caveat:** Axis A harmonizes beam *size* (4) but the fp16 (HF)
  and int8 (CTranslate2) engines differ in beam-search internals; treat small
  consistent deltas as "engine+quantization," and read Axis A as "is int8 ≥
  fp16," not "quantization improves quality."
- **COMET** needs the source text (it's reference+source based) — the pipeline
  passes it automatically. It downloads its model (`Unbabel/wmt22-comet-da`) on
  first use (internet).
