#!/bin/bash
# Fine-tune both affinity heads for five epochs using the top-N binary labels.
set -eo pipefail
TARGET="$1"; N="$2"
: "${WORKDIR:?}"; : "${BOLTZ_CACHE:?}"; : "${BOLTZ2_AFFINITY:?}"
PIPELINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$WORKDIR/runs/$TARGET"
CONSOL="$R/consolidated_full"
FTIN="$R/ft_inputs_full"
CFG="$BOLTZ2_AFFINITY/scripts/train/config_aff/config.yaml"
OUTDIR="$R/headft_full/top$N"
FTDIR="$R/headft_full/work_top$N"

echo "[train] target=$TARGET N=$N start=$(date)"
if [ -e "$FTDIR" ] || [ -e "$OUTDIR" ]; then
  echo "Training output already exists: $FTDIR or $OUTDIR" >&2; exit 2
fi
# Use the last 20 evaluation compounds for validation.
mkdir -p "$FTDIR"
tail -20 "$FTIN/eval_ids.txt" > "$FTDIR.val_ids.txt"
python "$PIPELINE/prepare_ft_dataset.py" \
  --ml-table "$FTIN/ml_table_train.csv" --target "$TARGET" \
  --source-yaml-dir "$CONSOL/yamls" --source-consolidated-dir "$CONSOL" \
  --out-dir "$FTDIR" \
  --train-ids-file "$FTIN/train_ids_top$N.txt" \
  --val-ids-file "$FTDIR.val_ids.txt"
SPLIT=$(ls "$FTDIR"/splits/val_explicit_train*.txt | head -1)
python "$BOLTZ2_AFFINITY/scripts/train/trainv2.py" "$CFG" \
  seed=0 \
  paths=example paths.pretrained="$BOLTZ_CACHE/boltz2_aff.ckpt" \
  paths.base_dir="$FTDIR" paths.mol_dir="$BOLTZ_CACHE/mols" paths.output_dir="$OUTDIR" \
  paths.split_file="$SPLIT" paths.samples_per_epoch="$N" \
  wandb=null save_top_k=1 disable_checkpoint=false \
  trainer.max_epochs=5 trainer.log_every_n_steps=5 \
  data.num_workers=4 data.random_seed=0 data.pad_to_max_tokens=false data.pad_to_max_atoms=false \
  trainer.accumulate_grad_batches=4 data.batch_size=1 \
  model.training_args.affinity_absolute_weight=0.0 \
  model.training_args.affinity_pairwise_weight=0.0 \
  model.training_args.affinity_binary_weight=1.0 \
  model.training_args.max_lr=0.00002 \
  model.training_args.lr_warmup_no_steps=2 \
  model.training_args.lr_start_decay_after_n_steps=20
CK=$(find "$OUTDIR" -name '*.ckpt' | head -1)
[ -z "$CK" ] && { echo "[train] NO CKPT for N=$N"; exit 1; }
echo "[train] done N=$N ckpt=$CK $(date)"
