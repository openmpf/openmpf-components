#!/usr/bin/env bash
# ===========================================================================
# Engine-vs-quantization decomposition for ONE language pair.
#
# Generates aligned per-sentence hypotheses for THREE systems, all beam 4,
# all per-line (batch 1 -> fair single-sentence-latency throughput + faithful
# quality), then scores the two decomposition contrasts:
#     engine effect      = HF-fp16  vs CT2-fp16   (same precision, diff engine)
#     quantization effect= CT2-fp16 vs CT2-int8   (same engine, diff precision)
# Produces decomp.SUMMARY.md with quality + throughput per system.
#
# Prereqs: ./convert_ct2.sh has produced models/nllb-3.3B-ct2-{float16,int8}.
# Usage:   ./run_decomp.sh              (default pair below)
#          PAIR="de-en|tmx/de-en.tmx|de|deu|Latn" N=2000 ./run_decomp.sh
# ===========================================================================
set -uo pipefail
EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EVAL"
PY=./venv/bin/python3

PAIR=${PAIR:-"pt-en|tmx/en-pt.tmx|pt|por|Latn"}
N=${N:-1000}                          # smaller than the quality pipeline: HF-fp16 per-line is slow
SEED=${SEED:-42}
GPU=${GPU:-'"device=0"'}
INT8_IMAGE_TAG="ctranslate2"
FP16_IMAGE_TAG="nllb-200-3.3B"
FP16_IMAGE=${FP16_IMAGE:-openmpf_nllb_translation:$FP16_IMAGE_TAG}
INT8_IMAGE=${INT8_IMAGE:-openmpf_nllb_translation:$INT8_IMAGE_TAG}
BOOTSTRAP=${BOOTSTRAP:-1000}
CT2_FP16=${CT2_FP16:-nllb-3.3B-ct2-float16}          # dir name under ./models
CT2_INT8=${CT2_INT8:-nllb-3.3B-ct2-int8_float16}     # int8_float16 = production-matching fast int8
# COMET on host -> CUDA_VISIBLE_DEVICES (see run_pipeline.sh notes)
_gpu_idx=$(printf '%s' "$GPU" | grep -oE '[0-9]+' | head -1)
COMET_DEVICE=${COMET_DEVICE:-${_gpu_idx:-0}}
if [ "$COMET_DEVICE" = "cpu" ]; then COMET_VISIBLE=""; COMET_GPUS=0; else COMET_VISIBLE="$COMET_DEVICE"; COMET_GPUS=1; fi

IFS='|' read -r name tmx tsrc nsrc nscript <<< "$PAIR"
H="$EVAL/results/decomp/$name"; RREL="results/decomp/$name"
mkdir -p "$H"
LOG="$H/decomp.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
nlines() { [ -f "$1" ] && wc -l < "$1" 2>/dev/null | tr -d ' ' || echo 0; }

log "DECOMP pair=$name ($nsrc"_"$nscript->eng_Latn) N=$N"

# model dirs must exist
for d in "$CT2_FP16" "$CT2_INT8"; do
  [ -f "models/$d/model.bin" ] || { log "MISSING models/$d/model.bin — run ./convert_ct2.sh first"; exit 1; }
done

# mteval/ct2_driver.py imports ctranslate2, which only exists in the CTranslate2 image.
# Check up front: otherwise a wrong INT8_IMAGE fails with ModuleNotFoundError
# only AFTER the slow HF-fp16 stage has already run.
if ! docker run --rm --entrypoint /opt/mpf/plugin-venv/bin/python "$INT8_IMAGE" \
       -c 'import ctranslate2' >/dev/null 2>&1; then
  log "INT8_IMAGE ($INT8_IMAGE) has no 'ctranslate2' module — mteval/ct2_driver.py cannot run in it."
  log "  Point INT8_IMAGE at a CTranslate2-based image (default: openmpf_nllb_translation:$INT8_IMAGE_TAG)."
  exit 1
fi

# sample
if [ ! -s "$H/sample.src" ]; then
  $PY -m mteval.tmx_sample --tmx "$tmx" --src-lang "$tsrc" -n "$N" --seed "$SEED" -o "$H/sample" >>"$LOG" 2>&1 \
    || { log "sample failed"; exit 1; }
fi
NL=$(nlines "$H/sample.src"); log "sample: $NL lines"

# --- generation (all per-line / batch 1) ---------------------------------
gen_component() {  # label image extra_mount extra_prop
  local label=$1 image=$2 mount=$3 prop=$4
  if [ "$(nlines "$H/hyp.$label.en")" -ge "$NL" ]; then log "$label already done"; return; fi
  log "generating $label (per-line, beam 4)..."
  local mnt=(); [ -n "$mount" ] && mnt=(-v "$mount")
  local extra=(); [ -n "$prop" ] && extra=(--prop "$prop")
  docker run --rm --gpus "$GPU" -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
    -v "$EVAL":/eval "${mnt[@]}" --entrypoint bash "$image" \
    -c "source /scripts/set-file-env-vars.sh 2>/dev/null || true; exec /opt/mpf/plugin-venv/bin/python /eval/mteval/nllb_eval_driver.py \
        --input /eval/$RREL/sample.src --output /eval/$RREL/hyp.$label.en \
        --source-lang $nsrc --source-script $nscript --num-beams 4 \
        ${extra[*]} --resume --progress-every 200 --meta-out /eval/$RREL/meta.$label.json" \
    >>"$LOG" 2>&1
  log "$label -> $(nlines "$H/hyp.$label.en")/$NL  ($(${PY} -c "import json;print(json.load(open('$H/meta.$label.json'))['sentences_per_sec'])" 2>/dev/null) sent/s)"
}

# CT2 systems: the component IGNORES NLLB_MODEL (loads its default and never
# reloads), so drive ctranslate2 directly with the standalone ct2_driver, which
# loads the mounted model and records the actual compute_type for verification.
gen_ct2() {  # label model_dir
  local label=$1 model=$2
  if [ "$(nlines "$H/hyp.$label.en")" -ge "$NL" ]; then log "$label already done"; return; fi
  log "generating $label via ct2_driver (model=$model)..."
  docker run --rm --gpus "$GPU" -v "$EVAL":/eval -v "$EVAL/models/$model:/models/$model" \
    --entrypoint /opt/mpf/plugin-venv/bin/python "$INT8_IMAGE" \
    /eval/mteval/ct2_driver.py --model "/models/$model" \
      --input /eval/$RREL/sample.src --output /eval/$RREL/hyp.$label.en \
      --source-lang "$nsrc" --source-script "$nscript" --beam 4 \
      --resume --progress-every 200 --meta-out /eval/$RREL/meta.$label.json \
    >>"$LOG" 2>&1
  local ct sp
  ct=$($PY -c "import json;print(json.load(open('$H/meta.$label.json'))['actual_compute_type'])" 2>/dev/null)
  sp=$($PY -c "import json;print(json.load(open('$H/meta.$label.json'))['sentences_per_sec'])" 2>/dev/null)
  log "$label -> $(nlines "$H/hyp.$label.en")/$NL  (compute_type=$ct, $sp sent/s)"
}

# DIFFICULT_LANGUAGE_TOKEN_LIMIT=0 is REQUIRED for parity, not optional. The
# develop component applies a 50-token "preferred limit" to languages it flags
# as difficult (Arabic by default), which sub-chunks even single sentences.
# mteval/ct2_driver.py has no such logic, so leaving it enabled makes the engine
# contrast measure chunking rather than the engine: with it on, ar-en showed a
# spurious +1.91 BLEU / +0.71 COMET "engine effect" (p<0.001) and HF-fp16 ran at
# ~55% the throughput of comparable pairs. run_pipeline.sh disables it for
# Axis A for the same reason.
gen_component hf-fp16  "$FP16_IMAGE" "" "DIFFICULT_LANGUAGE_TOKEN_LIMIT=0"
gen_ct2 ct2-fp16 "$CT2_FP16"
gen_ct2 ct2-int8 "$CT2_INT8"

# --- scoring: two contrasts ----------------------------------------------
score_pair() {  # a b tag
  log "scoring $3: $1 vs $2 (COMET dev=$COMET_DEVICE + bootstrap $BOOTSTRAP)..."
  CUDA_VISIBLE_DEVICES="$COMET_VISIBLE" $PY -m mteval.mt_eval compare \
    --hyp "$1"="$H/hyp.$1.en" --hyp "$2"="$H/hyp.$2.en" \
    -r "$H/sample.ref" -s "$H/sample.src" --comet --comet-gpus "$COMET_GPUS" \
    --bootstrap "$BOOTSTRAP" --csv "$H/decomp.$3.metrics.csv" \
    > "$H/decomp.$3.report.txt" 2>>"$LOG"
}
score_pair hf-fp16  ct2-fp16 engine
score_pair ct2-fp16 ct2-int8 quant

$PY -m mteval.decomp_report "$H" > "$H/decomp.SUMMARY.md" 2>>"$LOG" || true
log "DONE. Summary: $H/decomp.SUMMARY.md"
cat "$H/decomp.SUMMARY.md" 2>/dev/null || true
