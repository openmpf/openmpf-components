#!/usr/bin/env bash
# Readiness check — run before a big overnight job. No model loads, fast.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
INT8_IMAGE_TAG="ctranslate2"
FP16_IMAGE_TAG="nllb-200-3.3B"
INT8_IMAGE=${INT8_IMAGE:-openmpf_nllb_translation:$INT8_IMAGE_TAG}
FP16_IMAGE=${FP16_IMAGE:-openmpf_nllb_translation:$FP16_IMAGE_TAG}
ok(){ echo "  [OK] $*"; }; bad(){ echo "  [!!] $*"; FAIL=1; }
FAIL=0
echo "== preflight =="

command -v docker >/dev/null 2>&1 && ok "docker: $(docker --version)" || bad "docker not found"
command -v nvidia-smi >/dev/null 2>&1 && ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" \
  || bad "nvidia-smi not found (need NVIDIA runtime + GPU)"

for img in "$FP16_IMAGE" "$INT8_IMAGE"; do
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
  bad "venv missing — run ./setup_venv.sh"
fi

# Configured TMX files present? (skip commented-out example lines)
grep -vE '^[[:space:]]*#' run_pipeline.sh | grep -oE 'tmx/[^|"]+' | sort -u | while read -r f; do
  [ -f "$f" ] && ok "tmx present: $f" || bad "tmx MISSING: $f (drop it under ./tmx/)"
done

echo "== $([ "$FAIL" = 0 ] && echo 'READY' || echo 'NOT READY — fix [!!] items above') =="
exit "$FAIL"
