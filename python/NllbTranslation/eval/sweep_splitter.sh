#!/usr/bin/env bash
# ===========================================================================
# Splitter sweep — isolate WHY the token-based splitter under-generates on
# dense / non-Latin scripts in as-deployed (document) mode.
#
# Context: porting develop's token-based splitter fixed Chinese (ratio
# 0.551 -> 0.704) but regressed Bangla (0.825 -> 0.666) and Persian
# (0.899 -> 0.756). Axis A shows no per-sentence under-generation, so the loss
# is in chunking. This sweeps the levers Phase 5 exposed, on ONE pair, at a
# small sample size, so a cause can be found without re-running the pipeline.
#
# The decisive row is `char-control`: it reverts to the old character splitter
# on the NEW build. If it reproduces the old ratio, the splitter is the only
# variable and nothing else in Phases 1-7 contributed.
#
#   ./sweep_splitter.sh                       # bn-en, 400 sentences
#   PAIR=zh-en N=300 ./sweep_splitter.sh
#   CT2_IMAGE=openmpf_nllb_translation:ctranslate2-dev ./sweep_splitter.sh
# ===========================================================================
set -uo pipefail
EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EVAL"
PY=./venv/bin/python3

PAIR=${PAIR:-bn-en}
N=${N:-400}
GPU=${GPU:-'"device=0"'}
# Defaults match run_pipeline.sh. Locally the new build is :ctranslate2-dev;
# on a machine where the new build was tagged :ctranslate2, no override is needed.
CT2_IMAGE=${CT2_IMAGE:-${INT8_IMAGE:-openmpf_nllb_translation:ctranslate2}}

SRC="results/$PAIR/sample.src"
REF="results/$PAIR/sample.ref"
[ -s "$SRC" ] && [ -s "$REF" ] || { echo "missing $SRC / $REF — run the pipeline for $PAIR first"; exit 1; }

OUT="results/sweep/$PAIR"
mkdir -p "$OUT"
head -n "$N" "$SRC" > "$OUT/in.src"
head -n "$N" "$REF" > "$OUT/in.ref"

# Source language/script for the pair, taken from the pipeline's own table.
case "$PAIR" in
  bn-en) SLANG=ben; SSCRIPT=Beng ;;
  zh-en) SLANG=zho; SSCRIPT=Hans ;;
  fa-en) SLANG=pes; SSCRIPT=Arab ;;
  pt-en) SLANG=por; SSCRIPT=Latn ;;
  *) echo "unknown pair $PAIR — add it to the case block"; exit 1 ;;
esac

# name | extra -P job properties
CONFIGS=(
  "baseline|"
  "char-control|-P USE_NLLB_TOKEN_LENGTH=FALSE"
  "soft-250|-P NLLB_TRANSLATION_TOKEN_SOFT_LIMIT=250"
  "soft-400|-P NLLB_TRANSLATION_TOKEN_SOFT_LIMIT=400"
  "sentence-mode|-P SENTENCE_SPLITTER_MODE=SENTENCE"
  "len-penalty-1.5|-P NLLB_LENGTH_PENALTY=1.5"
)

echo "sweep: pair=$PAIR N=$N image=$CT2_IMAGE"
printf '%-16s %8s %8s %8s %9s %s\n' CONFIG BLEU RATIO CHUNKS OUT_CHARS NOTE

for entry in "${CONFIGS[@]}"; do
  name=${entry%%|*}; props=${entry#*|}
  json="$OUT/$name.json"; log="$OUT/$name.log"

  if [ ! -s "$json" ] || ! grep -q TRANSLATION "$json" 2>/dev/null; then
    # shellcheck disable=SC2086
    docker run --rm -i --gpus "$GPU" -e LOG_LEVEL=INFO "$CT2_IMAGE" \
      -t generic -P DEFAULT_SOURCE_SCRIPT="$SSCRIPT" -P DEFAULT_SOURCE_LANGUAGE="$SLANG" \
      $props - --pretty < "$OUT/in.src" > "$json" 2>"$log"
  fi

  if [ ! -s "$json" ] || ! grep -q TRANSLATION "$json" 2>/dev/null; then
    printf '%-16s %8s %8s %8s %9s %s\n' "$name" - - - - "FAILED (see $log)"
    continue
  fi

  # Chunk count is logged by the component; it distinguishes "chunks dropped"
  # from "chunks translated but truncated".
  chunks=$(grep -oE 'split into [0-9]+ chunks' "$log" | grep -oE '[0-9]+' | tail -1)
  [ -n "$chunks" ] || chunks="n/a"

  $PY mt_eval.py score "$json" -r "$OUT/in.ref" --csv "$OUT/$name.csv" > "$OUT/$name.report.txt" 2>>"$log"
  read -r bleu ratio chars <<<"$(
    $PY - "$OUT/$name.report.txt" <<'PYEOF'
import re, sys
t = open(sys.argv[1], encoding="utf-8").read()
b = re.search(r'BLEU\s+([0-9.]+)', t)
r = re.search(r'ratio=([0-9.]+)', t)
h = re.search(r'hyp=(\d+)', t)
print(b.group(1) if b else "-", r.group(1) if r else "-", h.group(1) if h else "-")
PYEOF
  )"
  printf '%-16s %8s %8s %8s %9s\n' "$name" "$bleu" "$ratio" "$chunks" "$chars"
done

echo
echo "artifacts in $OUT/  (delete a <config>.json to force that row to re-run)"
