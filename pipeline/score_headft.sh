#!/bin/bash
# Score a split with the pretrained or fine-tuned head on one shared cache.
# Usage: score_headft.sh base|headft TARGET N SPLIT_FILE OUT_CSV
set -eo pipefail
KIND="$1"; TARGET="$2"; N="$3"; A5="$4"; A6="$5"
: "${WORKDIR:?}"; : "${BOLTZ_CACHE:?}"
PIPELINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$WORKDIR/runs/$TARGET"
CACHE="$R/outputs_affcache"

if [ ! -d "$CACHE" ]; then
  echo "[score_headft] cache missing: $CACHE" >&2; exit 2
fi

if [ "$KIND" = "headft" ]; then
  CK="$R/headft_full/top${N}/checkpoints/last.ckpt"
  if [ ! -f "$CK" ]; then echo "[headft] ckpt missing: $CK" >&2; exit 1; fi
  echo "[headft] target=$TARGET N=$N chunk=$A5 -> $A6 start=$(date)"
  mkdir -p "$(dirname "$A6")"
  python "$PIPELINE/affinity_cached_infer.py" \
    --pretrained "$CK" --cache-dir "$CACHE" \
    --split-file "$A5" --out-csv "$A6" --no-kernels
  echo "[headft] done chunk=$A5 $(date)"

elif [ "$KIND" = "base" ]; then
  echo "[base] target=$TARGET chunk=$A5 -> $A6 start=$(date)"
  mkdir -p "$(dirname "$A6")"
  python "$PIPELINE/affinity_cached_infer.py" \
    --pretrained "$BOLTZ_CACHE/boltz2_aff.ckpt" --cache-dir "$CACHE" \
    --split-file "$A5" --out-csv "$A6" --no-kernels
  echo "[base] done chunk=$A5 $(date)"

else
  echo "unknown kind: $KIND" >&2; exit 1
fi
