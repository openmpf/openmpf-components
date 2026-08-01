#!/usr/bin/env bash
# Create the host-side scoring venv (needs internet for pip).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo "== verifying imports =="
./venv/bin/python - <<'PY'
import sacrebleu, comet, torch
print("sacrebleu", sacrebleu.__version__)
print("comet    ", getattr(comet, "__version__", "?"))
print("torch    ", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("  WARNING: torch does not see a GPU. COMET will run on CPU (slow),")
    print("  or install a torch build matching your CUDA (see requirements.txt).")
PY
echo "venv ready."
