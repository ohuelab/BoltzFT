"""Build comparator tables from the fixed compound IDs and Boltz score files."""
import argparse
import gzip
import json
import shutil
import os
from pathlib import Path
import pandas as pd
from verify_tables import digest


def scores(path, ids):
    files = sorted(path.glob('chunk_*.csv'))
    if not files:
        raise FileNotFoundError(path)
    df = pd.concat([pd.read_csv(p, dtype={'sample_id': str}) for p in files])
    if df.sample_id.duplicated().any():
        raise ValueError(f'Duplicate score IDs: {path}')
    return df.set_index('sample_id').affinity_probability_binary.reindex(ids).to_numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', type=Path, default=os.environ.get('WORKDIR'), required=False)
    p.add_argument('--out', type=Path, default=Path(os.environ['COMPARATOR_WORKDIR']) / 'tables')
    args = p.parse_args()
    if args.workdir is None:
        p.error('--workdir or WORKDIR is required')
    root = Path(__file__).resolve().parents[2]
    reference = json.loads((root / 'figures/data/baseline_cv/reference_digest.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    pockets = Path(os.environ['COMPARATOR_WORKDIR']) / 'pockets'
    pockets.mkdir(parents=True, exist_ok=True)
    for source in (root / 'data/drugclip_pockets').glob('*.npz'):
        shutil.copy2(source, pockets / source.name)
    for target, ref in reference['targets'].items():
        with gzip.open(root / 'data/splits' / f'{target}.txt.gz', 'rt') as f:
            ids = f.read().split()
        run = args.workdir / 'runs' / target
        df = pd.read_csv(run / 'ft_inputs_full/ml_table.csv', dtype={'complex_id': str})
        df = df.set_index('complex_id').loc[ids].reset_index()
        for n in [40, 100, 300]:
            train = set((run / 'ft_inputs_full' / f'train_ids_top{n}.txt').read_text().split())
            df[f'train_{n}'] = df.complex_id.isin(train)
        ev = set((run / 'ft_inputs_full/eval_ids.txt').read_text().split())
        df['is_eval'] = df.complex_id.isin(ev)
        if digest(df) != ref['sha256']:
            raise ValueError(f'{target}: labels or split membership differ from the paper')
        for col, folder in [('base', 'base')] + [(f'headft_{n}', f'lightning_top{n}') for n in [40, 100, 300]]:
            df[col] = scores(run / 'headft_affcache' / folder / 'scores', ids)
            if df.loc[df.is_eval, col].isna().any():
                raise ValueError(f'{target}/{col}: incomplete evaluation scores')
        df.to_csv(args.out / f'{target}.csv', index=False)
    (args.out / 'reference_digest.json').write_text(json.dumps(reference, indent=2) + '\n')


if __name__ == '__main__':
    main()
