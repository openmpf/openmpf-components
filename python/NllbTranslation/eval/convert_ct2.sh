#!/usr/bin/env bash
# Convert facebook/nllb-200-3.3B to CTranslate2 model directories (fp16 + int8),
# both from the SAME source checkpoint so the fp16-vs-int8 comparison is clean.
# Runs on the host venv (needs ctranslate2 + transformers + internet for the
# first download). Output dirs go under ./models/ and are mounted into the
# ctranslate2 image by run_decomp.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL=${MODEL:-facebook/nllb-200-3.3B}
REV=${REV:-1a07f7d195896b2114afcb79b7b57ab512e7b43e}   # pin for reproducibility
OUT=${OUT:-models}
CONV=./venv/bin/ct2-transformers-converter

if [ ! -x "$CONV" ]; then
  echo "ct2-transformers-converter not found in venv."
  echo "  ./venv/bin/pip install ctranslate2"
  exit 1
fi

# int8_float16 matches the production OpenNMT int8 model (int8 weights + fp16
# compute = the fast, slightly-lossy config). Plain `int8` defaults to
# int8_float32 (accurate but no tensor-core speedup) — NOT what we deploy.
QUANTS=(${QUANTS:-float16 int8_float16})
mkdir -p "$OUT"
for q in "${QUANTS[@]}"; do
  d="$OUT/nllb-3.3B-ct2-$q"
  if [ -f "$d/model.bin" ]; then
    echo "already converted: $d"
    continue
  fi
  echo "=== converting $MODEL ($REV) -> $d  [quantization=$q] ==="
  # --copy_files is load-bearing: without it the converted directory has no
  # tokenizer at all (only config.json, model.bin, shared_vocabulary.json), and
  # mteval/ct2_driver.py has nothing to encode with. The component's images hit exactly
  # this after the tokenizer download was dropped.
  "$CONV" --model "$MODEL" --revision "$REV" --output_dir "$d" \
    --quantization "$q" --low_cpu_mem_usage --force \
    --copy_files sentencepiece.bpe.model tokenizer.json \
                 tokenizer_config.json special_tokens_map.json
done
echo "done. CTranslate2 models under $OUT/  (tokenization uses the image's FLORES SPM)"
ls -la "$OUT"/nllb-3.3B-ct2-*/model.bin 2>/dev/null || true
