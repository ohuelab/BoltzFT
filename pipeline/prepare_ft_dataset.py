#!/usr/bin/env python3
"""Prepare binary-label training inputs from explicit training/validation IDs."""
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml


def read_ids(path):
    ids = path.read_text().split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate IDs: {path}")
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ml-table', type=Path, required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--source-yaml-dir', type=Path, required=True)
    p.add_argument('--source-consolidated-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--train-ids-file', type=Path, required=True)
    p.add_argument('--val-ids-file', type=Path, required=True)
    args = p.parse_args()
    frame = pd.read_csv(args.ml_table, dtype={'complex_id': str, 'target_id': str})
    frame = frame[(frame.dataset == 'MF-PCBA') & (frame.target_id == args.target)]
    if frame.complex_id.duplicated().any():
        raise ValueError('Duplicate complex IDs')
    train, val = read_ids(args.train_ids_file), read_ids(args.val_ids_file)
    keep = set(train) | set(val)
    if set(train) & set(val):
        raise ValueError('Training and validation IDs overlap')
    if keep - set(frame.complex_id):
        raise ValueError('Training or validation IDs missing from ml_table')
    frame = frame[frame.complex_id.isin(keep)]
    if not frame.is_binder.isin([0, 1]).all():
        raise ValueError('is_binder must contain binary labels')
    dst = args.out_dir / 'consolidated'
    yamls, splits = dst / 'yamls', args.out_dir / 'splits'
    yamls.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)
    for name in ['structures', 'records', 'msa', 'mols', 'embeddings']:
        src = args.source_consolidated_dir / name
        if not src.is_dir():
            raise FileNotFoundError(src)
        link = dst / name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve(), target_is_directory=True)
    manifest = json.loads((args.source_consolidated_dir / 'manifest.json').read_text())
    records = manifest['records'] if isinstance(manifest, dict) else manifest
    records = [x for x in records if str(x['id']) in keep]
    if {str(x['id']) for x in records} != keep:
        raise ValueError('Manifest does not cover the training and validation IDs')
    # The data module uses manifest minus validation IDs as its training set.
    filtered = {**manifest, 'records': records} if isinstance(manifest, dict) else records
    (dst / 'manifest.json').write_text(json.dumps(filtered))
    for row in frame.itertuples():
        doc = yaml.safe_load((args.source_yaml_dir / f'{row.complex_id}.yaml').read_text())
        ligand = next(x['ligand']['id'] for x in doc['sequences'] if 'ligand' in x)
        binder = ligand[0] if isinstance(ligand, list) else ligand
        props = [x for x in doc.get('properties', []) if 'affinity' not in x]
        props.append({'affinity': {'binder': binder, 'is_binder': int(row.is_binder),
                                   'assay_id': args.target, 'confidence': 1.0}})
        doc['properties'] = props
        (yamls / f'{row.complex_id}.yaml').write_text(yaml.safe_dump(doc, sort_keys=False))
    val_order = [cid for cid in frame.complex_id if cid in set(val)]
    (splits / f'val_explicit_train{len(train)}.txt').write_text('\n'.join(val_order) + '\n')
    print(f'{args.target}: {len(train)} training, {len(val)} validation compounds')


if __name__ == '__main__':
    main()
