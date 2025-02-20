#!/bin/bash

set -o errexit -o pipefail -o xtrace

/bin/ollama serve &

while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done

echo "Building LLaVA model..."
ollama pull llava:34b
echo "Done!"
