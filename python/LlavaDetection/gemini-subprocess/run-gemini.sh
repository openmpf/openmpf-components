#!/usr/bin/env bash

set -o errexit -o pipefail -o xtrace

usage() {
  echo "Usage: $(basename $0) [-m model] [-f <file_path>] [-p <prompt>] [-a <api_key>]"
  echo "  -m <model>      The name of the Gemini model to use"
  echo "  -f <file_path>  Path to the media file to process with Gemini"
  echo "  -p <prompt>     Path to image to process"
  echo "  -a <api_key>    Google API key"
  echo ""
  echo "Example:"
  echo "  $(basename $0) -f /mypath/myfile.png -p \"Describe this image\" -a abcdefghijkl"
}

while getopts 'hm:f:p:a:' opt; do
  case "$opt" in
    m)
      MODEL="$OPTARG"
      ;;

    f)
      FILE_PATH="$OPTARG"
      ;;

    p)
      PROMPT="$OPTARG"
      ;;

    a)
      API_KEY="$OPTARG"
      ;;

    ?|h)
      usage
      exit 1
      ;;
  esac
done

# Mandatory arguments
if [ ! "$FILE_PATH" ] || [ ! "$PROMPT" ] || [ ! "$API_KEY" ]; then
  echo "arguments -f, -p, and -a must be provided"
  usage
  exit 1
fi

source /gemini-subprocess/venv/bin/activate

if [ "$MODEL" ]; then
    python3 /gemini-subprocess/gemini-process-image.py -m "$MODEL" -f "$FILE_PATH" -p "$PROMPT" -a "$API_KEY"
else
    python3 /gemini-subprocess/gemini-process-image.py -f "$FILE_PATH" -p "$PROMPT" -a "$API_KEY"
fi
