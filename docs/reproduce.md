# Usage

## Setup

Download `mf-pcba_test.zip` from the
[boltzina v1.0.1 release](https://github.com/ohuelab/boltzina/releases/tag/v1.0.1).
For each target, use its results CSV (`CID`, `neut-smiles`, `Active_v2`, and
`affinity_probability_binary`) and the construct/MSA from the initial Boltz
run. Compound IDs are listed in `data/splits/`.

Install the [Boltz2_affinity](https://github.com/molecularinformatics/Boltz2_affinity)
environment and apply the patch:

```bash
export BOLTZFT=/path/to/BoltzFT
export BOLTZ2_AFFINITY=/path/to/Boltz2_affinity
git clone https://github.com/molecularinformatics/Boltz2_affinity.git "$BOLTZ2_AFFINITY"
git -C "$BOLTZ2_AFFINITY" checkout bc06a0b
git -C "$BOLTZ2_AFFINITY" apply "$BOLTZFT/patches/boltz2_affinity.patch"
python -m pip install -r "$BOLTZFT/requirements.txt"
```

Obtain the Boltz-2 weights and CCD molecule directory following the fork's
setup instructions, then set the paths:

```bash
export WORKDIR=/path/to/results
export BOLTZ_CACHE=/path/to/boltz-cache
export PYTHONPATH="$BOLTZ2_AFFINITY/src${PYTHONPATH:+:$PYTHONPATH}"
export TARGET=588689
export RESULTS_CSV=/path/to/588689_results.csv
export MSA_CSV=/path/to/588689_msa.csv
R="$WORKDIR/runs/$TARGET"
```

## Prepare inputs

```bash
python "$BOLTZFT/pipeline/select_subsets.py" \
  --target "$TARGET" --results-csv "$RESULTS_CSV" --out-dir "$R/selection"
python "$BOLTZFT/pipeline/build_boltz_inputs.py" \
  --target "$TARGET" --inference-csv "$R/selection/inference_compounds.csv" \
  --msa-csv "$MSA_CSV" --out-yaml-dir "$R/inputs_full"
python "$BOLTZFT/pipeline/make_chunks.py" \
  --target "$TARGET" --inputs-dir "$R/inputs_full" \
  --chunks-root "$R/inputs_chunks_full" --outputs-root "$R/outputs_chunks_full" \
  --index-file "$R/chunks.tsv"

for chunk in "$R"/inputs_chunks_full/chunk_*; do
  boltz predict "$chunk" --out_dir "$R/outputs_chunks_full/$(basename "$chunk")" \
    --cache "$BOLTZ_CACHE" --model boltz2 --devices 1 --accelerator gpu \
    --num_workers 2 --preprocessing-threads 1 \
    --write_embeddings_cropped --skip_affinity_prediction
done

python "$BOLTZFT/pipeline/merge_consolidate.py" \
  --outputs-root "$R/outputs_chunks_full" --inputs-dir "$R/inputs_full" \
  --out-dir "$R/consolidated_full"
python "$BOLTZFT/pipeline/build_ft_inputs_full.py" \
  --target "$TARGET" --results-csv "$RESULTS_CSV" \
  --consolidated-dir "$R/consolidated_full" --out-dir "$R/ft_inputs_full"
```

## Training and scoring

```bash
for chunk in "$R"/inputs_chunks_full/chunk_*; do
  bash "$BOLTZFT/pipeline/build_affinity_cache.sh" "$TARGET" "$chunk" \
    "$R/outputs_affcache/$(basename "$chunk")"
done
for n in 40 100 300; do
  bash "$BOLTZFT/pipeline/train_headft.sh" "$TARGET" "$n"
done
for split in "$R"/ft_inputs_full/eval_chunks/chunk_*.txt; do
  name="$(basename "$split" .txt)"
  bash "$BOLTZFT/pipeline/score_headft.sh" base "$TARGET" 0 "$split" \
    "$R/headft_affcache/base/scores/$name.csv"
  for n in 40 100 300; do
    bash "$BOLTZFT/pipeline/score_headft.sh" headft "$TARGET" "$n" "$split" \
      "$R/headft_affcache/lightning_top$n/scores/$name.csv"
  done
done
```

## Evaluation

```bash
python "$BOLTZFT/evaluation/eval_headft_cached.py" --targets "$TARGET" \
  --out-csv "$WORKDIR/metrics_$TARGET.csv"
python "$BOLTZFT/evaluation/eval_cascade.py" --targets "$TARGET" \
  --out-csv "$WORKDIR/cascade_$TARGET.csv"
```

## Comparison methods

```bash
export COMPARATOR_WORKDIR="$WORKDIR/comparators"
export FEATURE_CACHE_DIR="$WORKDIR/features"
python "$BOLTZFT/evaluation/drugclip/prepare_tables.py"
```

Build features for each target in the patched Boltz environment.
Install `chemprop>=2.2.0` for CheMeleon.

```bash
python "$BOLTZFT/evaluation/extract_exact_graw.py" \
  --ml-table "$R/ft_inputs_full/ml_table.csv" \
  --embeddings-root "$R/outputs_affcache" \
  --out "$FEATURE_CACHE_DIR/graw/$TARGET.npz"
python "$BOLTZFT/evaluation/make_exact_chemeleon_cache.py" \
  --results-csv "$RESULTS_CSV" --ml-table "$R/ft_inputs_full/ml_table.csv" \
  --out "$FEATURE_CACHE_DIR/chemeleon/$TARGET.npz"
python "$BOLTZFT/evaluation/drugclip/supervised_baselines.py" --targets "$TARGET" \
  --out "$COMPARATOR_WORKDIR/lightgbm_$TARGET.csv"
```

Install [DrugCLIP](https://github.com/THU-ATOM/DrugCLIP) and its dependencies,
then download the model weights following its setup instructions.
Set the source and checkpoint paths:

```bash
export DRUGCLIP_ROOT=/path/to/DrugCLIP
export DRUGCLIP_CHECKPOINT=/path/to/checkpoint_best.pt
```

Run the following after preparing input tables for all targets:

```bash
python "$BOLTZFT/evaluation/drugclip/build_smiles_index.py"
python "$BOLTZFT/evaluation/drugclip/gen_confs.py"
python "$BOLTZFT/evaluation/drugclip/encode_mols.py"
python "$BOLTZFT/evaluation/drugclip/collect_embs.py"
python "$BOLTZFT/evaluation/drugclip/official_zeroshot.py"
python "$BOLTZFT/evaluation/drugclip/run_eval.py"
```
