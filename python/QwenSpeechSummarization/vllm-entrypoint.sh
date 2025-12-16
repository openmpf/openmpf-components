#!/bin/bash -e

set -o pipefail

model_string="$(echo "${VLLM_MODEL}" | sed 's/\//--/g')" # replace / with --
snapshot_glob="/root/.cache/huggingface/hub/models--${model_string}/snapshots/*/"

for x in $snapshot_glob; do
    vllm serve $x --served-model-name "${VLLM_MODEL}" "$@" || continue
    exit 0
done
echo "Failed to find a valid snapshot directory for the model" 1>&2
exit 1