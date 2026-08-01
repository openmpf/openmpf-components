#!/usr/bin/env bash
# ===========================================================================
# Fetch the TMX corpora the pipeline needs, into ./tmx/.
#
# Idempotent: an already-extracted .tmx is left alone, a stray .gz is just
# decompressed, and only what is genuinely missing is downloaded. Safe to
# re-run after an interrupted download.
#
# The file list is derived from the PAIRS table in run_pipeline.sh -- the same
# source preflight.sh checks against -- so adding a pair there is enough and
# the two cannot drift. Corpora are OPUS TED2020 v1.
#
#   ./download_tmx_files.sh          # fetch whatever is missing
#   ./download_tmx_files.sh -n       # report what would be fetched, download nothing
# ===========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_URL=${BASE_URL:-https://object.pouta.csc.fi/OPUS-TED2020/v1/tmx}
DEST=tmx
DRY=0
[ "${1:-}" = "-n" ] && DRY=1

command -v wget >/dev/null || { echo "wget not found; install it or set up ./tmx manually." >&2; exit 1; }

# Same extraction preflight.sh uses: skip commented lines, pull the tmx/ paths
# out of the PAIRS table. Falls back to the known nine if that ever yields
# nothing, so a reformat of run_pipeline.sh cannot silently produce a no-op.
mapfile -t WANTED < <(grep -vE '^[[:space:]]*#' run_pipeline.sh \
                      | grep -oE 'tmx/[^|"]+\.tmx' | sort -u)
if [ "${#WANTED[@]}" -eq 0 ]; then
  echo "WARNING: could not read the PAIRS table in run_pipeline.sh; using the built-in list." >&2
  WANTED=(tmx/ar-en.tmx tmx/bn-en.tmx tmx/de-en.tmx tmx/en-fa.tmx tmx/en-fr.tmx
          tmx/en-pt.tmx tmx/en-ru.tmx tmx/en-uk.tmx tmx/en-zh_cn.tmx)
fi

mkdir -p "$DEST"
have=0 fetched=0 unzipped=0 failed=0

for path in "${WANTED[@]}"; do
  name=$(basename "$path")            # e.g. en-zh_cn.tmx
  if [ -s "$DEST/$name" ]; then
    echo "  have      $name"
    have=$((have + 1))
    continue
  fi

  if [ -s "$DEST/$name.gz" ]; then
    echo "  unzip     $name.gz (already downloaded)"
    [ "$DRY" = 1 ] || gunzip -f "$DEST/$name.gz"
    unzipped=$((unzipped + 1))
    continue
  fi

  if [ "$DRY" = 1 ]; then
    echo "  would get $name"
    fetched=$((fetched + 1))
    continue
  fi

  echo "  download  $name"
  # Download to a .part file so an interrupted transfer is never mistaken for a
  # complete .gz on the next run.
  if wget -q --show-progress -O "$DEST/$name.gz.part" "$BASE_URL/$name.gz"; then
    mv "$DEST/$name.gz.part" "$DEST/$name.gz"
    gunzip -f "$DEST/$name.gz"
    fetched=$((fetched + 1))
  else
    rm -f "$DEST/$name.gz.part"
    echo "  FAILED    $name  ($BASE_URL/$name.gz)" >&2
    failed=$((failed + 1))
  fi
done

echo
echo "present: $have   downloaded: $fetched   extracted from .gz: $unzipped   failed: $failed"
if [ "$failed" -gt 0 ]; then
  echo "Some corpora are missing. If a pair is not on OPUS TED2020, drop its TMX into ./$DEST" >&2
  echo "by hand, or remove that pair from the PAIRS table in run_pipeline.sh." >&2
  exit 1
fi
[ "$DRY" = 1 ] && echo "(dry run -- nothing was written)"
echo "Next: ./preflight.sh"
