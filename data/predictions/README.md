# Compound-level predictions

`<target>.csv.gz` contains compound information, assay labels and prediction
scores for each of the eight MF-PCBA targets. All methods use a common
evaluation set excluding the 300 training candidates. No-FT scores cover all
compounds; other scores are provided for the evaluation set.

## Columns

- `target`, `complex_id`, `ligand_id`, `smiles`: target and compound identifiers.
- `is_binder`: experimental assay label (1 active, 0 inactive).
- `train_40`, `train_100`, `train_300`, `is_eval`: training/evaluation membership.
- `eval_order`: original evaluation order; -1 for training candidates.
- `no_ft`: No-FT score.
- `headft_N{40,100,300}`: Head-FT score at each training budget.
- `{ecfp,graw,chemeleon,ecfp_graw,chemeleon_graw}_lgbm_cv_N{40,100,300}`:
  CV-selected LightGBM score. `graw` denotes the Boltz trunk features.
- `drugclip_zeroshot`: DrugCLIP zero-shot score.
- `drugclip_headft_N{40,100,300}_seed{0,1,2,3,4}`: DrugCLIP fine-tuning scores.

Higher scores rank compounds earlier. LightGBM scores average five models;
DrugCLIP fine-tuning results average the metrics calculated separately for its
five runs. CV-based scores at N=40 are unavailable for `493248-485317` and
`1053173-743445` and are left empty.

## Recalculate results

Run from the repository root:

```bash
uv run --no-project --python 3.11 --with-requirements evaluation/prediction_requirements.txt \
  python evaluation/recompute_predictions.py
```

Results are written to `runs/prediction_verification/` and compared with the
supplied reference metrics. The script preserves `eval_order` for tied scores.
