#!/usr/bin/env bash
# ===========================================================================
# End-to-end MT quality eval pipeline for multiple language pairs.
# Sample from TMX -> generate Axis A (per-sentence, fp16 & int8) and Axis B
# (as-deployed blob) -> score (BLEU/chrF/chrF++/TER + COMET + paired bootstrap)
# -> per-pair reports + combined SUMMARY. Sequential (shared GPU), resumable.
#
# Portable: resolves its own directory, so it runs wherever this folder lives.
# Requires: Docker + NVIDIA runtime, the two OpenMPF NllbTranslation images
# (see README), and the local venv (./setup_venv.sh).
#
#   ./run_pipeline.sh                 # all configured pairs
#   ./run_pipeline.sh ar-en           # one pair by name
#   RUN_AXIS_B=0 ./run_pipeline.sh    # skip slow as-deployed blobs (Axis A only)
# ===========================================================================
set -uo pipefail
EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this folder, wherever it is
cd "$EVAL"
PY=./venv/bin/python3

# ----- config: EDIT THIS for your data -------------------------------------
# One line per pair:  name | tmx_path | tmx_src_lang | nllb_src_lang | nllb_src_script
#   name           short label + output dir (results/<name>/)
#   tmx_path        path to the TMX (drop yours under ./tmx/)
#   tmx_src_lang    the xml:lang of the SOURCE side in the TMX (e.g. ar, zh_cn, de)
#   nllb_src_lang   NLLB/FLORES-200 language part  (arb, zho, deu, fra, spa, rus ...)
#   nllb_src_script NLLB/FLORES-200 script part    (Arab, Hans, Latn, Cyrl ...)
# Target is always English (eng_Latn). See README for the code lookup.
PAIRS=(
  "ar-en|tmx/ar-en.tmx|ar|arb|Arab"        # Arabic     -> English
  "bn-en|tmx/bn-en.tmx|bn|ben|Beng"        # Bangla     -> English
  "de-en|tmx/de-en.tmx|de|deu|Latn"        # German     -> English
  "fa-en|tmx/en-fa.tmx|fa|pes|Arab"        # Persian (Western) -> English
  "fr-en|tmx/en-fr.tmx|fr|fra|Latn"        # French     -> English
  "pt-en|tmx/en-pt.tmx|pt|por|Latn"        # Portuguese -> English
  "ru-en|tmx/en-ru.tmx|ru|rus|Cyrl"        # Russian    -> English
  "uk-en|tmx/en-uk.tmx|uk|ukr|Cyrl"        # Ukrainian  -> English
  "zh-en|tmx/en-zh_cn.tmx|zh_cn|zho|Hans"  # Chinese (Simplified) -> English
)

N=${N:-5000}                          # sentences per pair
SEED=${SEED:-42}
GPU=${GPU:-'"device=0"'}              # e.g. '"device=1"' for another GPU
INT8_IMAGE_TAG="ctranslate2"
FP16_IMAGE_TAG="nllb-200-3.3B"
INT8_IMAGE=${INT8_IMAGE:-openmpf_nllb_translation:$INT8_IMAGE_TAG}
FP16_IMAGE=${FP16_IMAGE:-openmpf_nllb_translation:$FP16_IMAGE_TAG}
RUN_AXIS_B=${RUN_AXIS_B:-1}           # 0 = skip slow as-deployed blob runs
BOOTSTRAP=${BOOTSTRAP:-1000}
FP16_BATCH=${FP16_BATCH:-16}          # lower if the GPU has <16 GB

# COMET scoring runs on the HOST (not in Docker), so it selects its GPU via
# CUDA_VISIBLE_DEVICES, NOT the --gpus flag above. Default it to the same GPU
# index as $GPU so one setting covers both. Override with COMET_DEVICE=<idx>
# or COMET_DEVICE=cpu.
_gpu_idx=$(printf '%s' "$GPU" | grep -oE '[0-9]+' | head -1)
COMET_DEVICE=${COMET_DEVICE:-${_gpu_idx:-0}}
if [ "$COMET_DEVICE" = "cpu" ]; then COMET_VISIBLE=""; COMET_GPUS=0
else COMET_VISIBLE="$COMET_DEVICE"; COMET_GPUS=1; fi
# ---------------------------------------------------------------------------

RUNLOG="$EVAL/results/pipeline.$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$EVAL/results"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }
nlines() { wc -l < "$1" 2>/dev/null | tr -d ' ' || echo 0; }

driver_run() {  # image env_args... -- driver_args...
  local image=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  docker run --rm --gpus "$GPU" "${envs[@]}" -v "$EVAL":/eval \
    --entrypoint bash "$image" \
    -c "source /scripts/set-file-env-vars.sh 2>/dev/null || true; exec /opt/mpf/plugin-venv/bin/python /eval/nllb_eval_driver.py $*"
}

blob_run() {  # image script lang outjson srcfile
  docker run --rm -i --gpus "$GPU" -e LOG_LEVEL=INFO "$1" \
    -t generic -P DEFAULT_SOURCE_SCRIPT="$2" -P DEFAULT_SOURCE_LANGUAGE="$3" - --pretty \
    < "$5" > "$4"
}

run_pair() {
  local name=$1 tmx=$2 tsrc=$3 nsrc=$4 nscript=$5
  local RREL="results/$name" H="$EVAL/results/$name"
  mkdir -p "$H"
  log "=================== PAIR $name  ($nsrc"_"$nscript -> eng_Latn) ==================="

  if [ ! -s "$H/sample.src" ]; then
    log "$name: sampling $N pairs from $tmx"
    $PY tmx_sample.py --tmx "$tmx" --src-lang "$tsrc" -n "$N" --seed "$SEED" \
        -o "$H/sample" >>"$RUNLOG" 2>&1 || { log "$name: SAMPLE FAILED, skipping"; return; }
  fi
  local NL; NL=$(nlines "$H/sample.src")
  log "$name: sample has $NL lines"
  [ "$NL" -gt 0 ] || { log "$name: empty sample, skipping"; return; }

  log "$name: smoke test (int8, 3 lines) to validate lang code '$nsrc"_"$nscript'"
  driver_run "$INT8_IMAGE" -- \
    --input /eval/$RREL/sample.src --output /eval/$RREL/smoke.int8.en \
    --source-lang "$nsrc" --source-script "$nscript" --num-beams 4 --limit 3 >>"$RUNLOG" 2>&1
  if [ "$(nlines "$H/smoke.int8.en")" -lt 3 ]; then
    log "$name: SMOKE FAILED (lang code likely rejected) — skipping pair"; return
  fi
  rm -f "$H/smoke.int8.en"; log "$name: smoke OK"

  if [ "$(nlines "$H/hyp.int8.en")" -ge "$NL" ]; then
    log "$name: Axis A int8 already complete"
  else
    log "$name: Axis A int8 generating..."
    driver_run "$INT8_IMAGE" -- \
      --input /eval/$RREL/sample.src --output /eval/$RREL/hyp.int8.en \
      --source-lang "$nsrc" --source-script "$nscript" --num-beams 4 \
      --resume --progress-every 500 --meta-out /eval/$RREL/meta.int8.json >>"$RUNLOG" 2>&1
    log "$name: Axis A int8 -> $(nlines "$H/hyp.int8.en")/$NL"
  fi

  if [ "$(nlines "$H/hyp.fp16.en")" -ge "$NL" ]; then
    log "$name: Axis A fp16 already complete"
  else
    log "$name: Axis A fp16 generating (batch $FP16_BATCH, OOM-safe)..."
    driver_run "$FP16_IMAGE" -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 -- \
      --input /eval/$RREL/sample.src --output /eval/$RREL/hyp.fp16.en \
      --source-lang "$nsrc" --source-script "$nscript" --num-beams 4 \
      --batch "$FP16_BATCH" --gpu-empty-cache-every 5 --prop DIFFICULT_LANGUAGE_TOKEN_LIMIT=0 \
      --resume --progress-every 500 --meta-out /eval/$RREL/meta.fp16.json >>"$RUNLOG" 2>&1
    log "$name: Axis A fp16 -> $(nlines "$H/hyp.fp16.en")/$NL"
  fi

  if [ "$RUN_AXIS_B" = "1" ]; then
    for m in int8 fp16; do
      local img=$INT8_IMAGE; [ "$m" = fp16 ] && img=$FP16_IMAGE
      if [ -s "$H/asdeployed.$m.json" ] && grep -q TRANSLATION "$H/asdeployed.$m.json" 2>/dev/null; then
        log "$name: Axis B $m blob already done"
      else
        log "$name: Axis B $m blob generating (slow, whole-doc as-shipped)..."
        blob_run "$img" "$nscript" "$nsrc" "$H/asdeployed.$m.json" "$H/sample.src" >>"$RUNLOG" 2>&1
        log "$name: Axis B $m blob -> $(wc -c < "$H/asdeployed.$m.json" 2>/dev/null) bytes"
      fi
    done
  else
    log "$name: RUN_AXIS_B=0, skipping as-deployed blobs"
  fi

  if [ "$(nlines "$H/hyp.fp16.en")" -ge "$NL" ] && [ "$(nlines "$H/hyp.int8.en")" -ge "$NL" ]; then
    log "$name: scoring Axis A (COMET on device '$COMET_DEVICE' + bootstrap $BOOTSTRAP)..."
    CUDA_VISIBLE_DEVICES="$COMET_VISIBLE" $PY mt_eval.py compare \
      --hyp fp16="$H/hyp.fp16.en" --hyp int8="$H/hyp.int8.en" \
      -r "$H/sample.ref" -s "$H/sample.src" --comet --comet-gpus "$COMET_GPUS" \
      --bootstrap "$BOOTSTRAP" \
      --per-segment "$H/axisA.segments.csv" --csv "$H/axisA.metrics.csv" \
      > "$H/axisA.report.txt" 2>>"$RUNLOG"
    log "$name: Axis A report -> $H/axisA.report.txt"
  else
    log "$name: Axis A incomplete, skipping scoring"
  fi

  if [ "$RUN_AXIS_B" = "1" ]; then
    for m in fp16 int8; do
      [ -s "$H/asdeployed.$m.json" ] && $PY mt_eval.py score "$H/asdeployed.$m.json" \
        -r "$H/sample.ref" --csv "$H/axisB.$m.csv" > "$H/axisB.$m.report.txt" 2>>"$RUNLOG"
    done
    log "$name: Axis B reports written"
  fi
  log "$name: DONE"
}

log "PIPELINE START (pairs=${#PAIRS[@]}, N=$N, RUN_AXIS_B=$RUN_AXIS_B, translate GPU=$GPU, COMET device=$COMET_DEVICE)"
WANT="${1:-}"
for entry in "${PAIRS[@]}"; do
  IFS='|' read -r name tmx tsrc nsrc nscript <<< "$entry"
  [ -n "$WANT" ] && [ "$WANT" != "$name" ] && continue
  run_pair "$name" "$tmx" "$tsrc" "$nsrc" "$nscript"
done
log "assembling combined SUMMARY..."
$PY summarize.py >> "$RUNLOG" 2>&1 || true
log "PIPELINE COMPLETE. Log: $RUNLOG   Summary: results/SUMMARY.md"
