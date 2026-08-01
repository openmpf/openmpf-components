#!/usr/bin/env bash
# Readiness check — run before a big overnight job. No model loads, fast.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# CT2 = CTranslate2-branch image, HF = develop/Transformers image. Old INT8_/FP16_
# names still accepted; see 03_run_pipeline.sh.
CT2_IMAGE_TAG="ctranslate2"
HF_IMAGE_TAG="nllb-200-3.3B"
CT2_IMAGE=${CT2_IMAGE:-${INT8_IMAGE:-openmpf_nllb_translation:$CT2_IMAGE_TAG}}
HF_IMAGE=${HF_IMAGE:-${FP16_IMAGE:-openmpf_nllb_translation:$HF_IMAGE_TAG}}
ok(){ echo "  [OK] $*"; }; bad(){ echo "  [!!] $*"; FAIL=1; }
FAIL=0
echo "== preflight =="

command -v docker >/dev/null 2>&1 && ok "docker: $(docker --version)" || bad "docker not found"
command -v nvidia-smi >/dev/null 2>&1 && ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" \
  || bad "nvidia-smi not found (need NVIDIA runtime + GPU)"

for img in "$HF_IMAGE" "$CT2_IMAGE"; do
  docker image inspect "$img" >/dev/null 2>&1 && ok "image present: $img" \
    || bad "image MISSING: $img  (build it — see README)"
done

if [ -x ./venv/bin/python ]; then
  ./venv/bin/python - <<'PY' && ok "venv imports OK" || bad "venv import failure"
import sacrebleu, comet, torch  # noqa
PY
  cuda=$(./venv/bin/python -c "import torch;print(torch.cuda.is_available())" 2>/dev/null)
  [ "$cuda" = "True" ] && ok "COMET can use GPU" || echo "  [~~] COMET GPU unavailable -> will use CPU (slow)"
else
  bad "venv missing — run ./00_setup_venv.sh"
fi

# Configured TMX files present? (skip commented-out example lines)
# Read via process substitution, NOT a pipeline: `... | while read` puts the loop in
# a subshell, so bad()'s FAIL=1 is discarded and a missing corpus still reports READY.
PIPELINE=03_run_pipeline.sh
if [ ! -f "$PIPELINE" ]; then
  bad "$PIPELINE not found — cannot check the configured TMX files"
else
  npairs=0
  while read -r f; do
    npairs=$((npairs + 1))
    [ -f "$f" ] && ok "tmx present: $f" || bad "tmx MISSING: $f (drop it under ./tmx/)"
  done < <(grep -vE '^[[:space:]]*#' "$PIPELINE" | grep -oE 'tmx/[^|"]+' | sort -u)
  # Zero matches means the PAIRS table moved or was reformatted. Without this the
  # loop just does nothing and preflight passes having verified no corpora at all.
  [ "$npairs" -gt 0 ] || bad "no tmx/ paths found in $PIPELINE — has the PAIRS table changed?"
fi

echo "== $([ "$FAIL" = 0 ] && echo 'READY' || echo 'NOT READY — fix [!!] items above') =="
exit "$FAIL"
