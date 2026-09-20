#!/usr/bin/env python3
"""Prepare compound tables from the supplied ranked IDs."""
import argparse
import gzip
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True)
    p.add_argument('--results-csv', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    df = pd.read_csv(args.results_csv)
    cols = ['CID', 'neut-smiles', 'Active_v2', 'affinity_probability_binary']
    if not set(cols) <= set(df.columns):
        raise ValueError(f'Required columns: {cols}')
    df = df.dropna(subset=['affinity_probability_binary', 'neut-smiles']).copy()
    df.CID = df.CID.astype(int)
    df = df.drop_duplicates('CID', keep='first')
    if not df.Active_v2.isin([0, 1]).all() or len(df) <= 320:
        raise ValueError('Expected a binary-labelled library with more than 320 compounds')
    ids_path = Path(__file__).resolve().parents[1] / 'data/splits' / f'{args.target}.txt.gz'
    with gzip.open(ids_path, 'rt') as f:
        ids = [int(cid.strip().removeprefix(args.target + '_')) for cid in f]
    df = df.set_index('CID').loc[ids].reset_index()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / 'inference_compounds.csv', index=False)
    for n in [40, 100, 300]:
        df.head(n).to_csv(args.out_dir / f'train_top{n}.csv', index=False)
    df.iloc[300:].to_csv(args.out_dir / 'eval_subset.csv', index=False)


if __name__ == '__main__':
    main()
