#!/bin/bash
# Build affinity-checkpoint caches using the stored structure-stage poses.
# Usage: build_affinity_cache.sh TARGET CHUNK_DIR OUT_DIR
set -eo pipefail
TARGET="$1"; CHUNK_DIR="$2"; OUT_DIR="$3"
: "${BOLTZ_CACHE:?BOLTZ_CACHE not set}"
STRUCT_DIR="${STRUCT_DIR:-${WORKDIR:?WORKDIR or STRUCT_DIR must be set}/runs/$TARGET/consolidated_full/structures}"
if [ ! -d "$STRUCT_DIR" ]; then
  echo "[build_affinity_cache] STRUCT_DIR not found: $STRUCT_DIR" >&2
  exit 2
fi
# KERNELS=1 (default on a full GPU) enables triton kernels; set KERNELS=0 on MIG.
KERNEL_FLAG="--no_kernels"; [ "${KERNELS:-1}" = "1" ] && KERNEL_FLAG=""
mkdir -p "$OUT_DIR"
echo "[build_affinity_cache] target=$TARGET chunk=$CHUNK_DIR -> $OUT_DIR kernels=${KERNELS:-1} start=$(date)"
echo "[build_affinity_cache] reuse_poses=$STRUCT_DIR"
echo "[build_affinity_cache] yaml_count=$(find -L "$CHUNK_DIR" -maxdepth 1 -name '*.yaml' | wc -l)"
boltz predict "$CHUNK_DIR" \
  --out_dir "$OUT_DIR" \
  --cache "$BOLTZ_CACHE" \
  --devices 1 --accelerator gpu --model boltz2 \
  --num_workers 2 --preprocessing-threads 1 \
  $KERNEL_FLAG --write_affinity_embeddings_cropped \
  --affinity_skip_run_structure --diffusion_samples_affinity 5 \
  --reuse_pre_affinity_dir "$STRUCT_DIR" --override
echo "[build_affinity_cache] done=$(date) affinity_crop_emb=$(find -L "$OUT_DIR" -name 'affinity_embeddings_*.npz' | wc -l)"
