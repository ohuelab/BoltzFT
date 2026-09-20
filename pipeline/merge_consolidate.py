#!/usr/bin/env python3
"""Merge chunked Boltz outputs for one target into a single training-ready consolidated tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", type=Path, required=True, help="runs/<target>/outputs_chunks")
    p.add_argument("--inputs-dir", type=Path, required=True, help="runs/<target>/inputs (all yamls)")
    p.add_argument("--out-dir", type=Path, required=True, help="runs/<target>/consolidated")
    args = p.parse_args()

    results_dirs = sorted(args.outputs_root.glob("chunk_*/boltz_results_*"))
    if not results_dirs:
        raise SystemExit(f"no boltz_results_* under {args.outputs_root}/chunk_*/")

    emb_dir = args.out_dir / "embeddings"
    struct_dir = args.out_dir / "structures"
    rec_dir = args.out_dir / "records"
    mol_dir = args.out_dir / "mols"
    msa_dir = args.out_dir / "msa"
    for d in (emb_dir, struct_dir, rec_dir, mol_dir, msa_dir):
        d.mkdir(parents=True, exist_ok=True)

    records = []
    n_emb = n_struct = n_msa = 0
    for rd in results_dirs:
        processed = rd / "processed"
        predictions = rd / "predictions"
        for emb in predictions.glob("*/embeddings_*.npz"):
            link(emb, emb_dir / emb.name); n_emb += 1
        for npz in (processed / "structures").glob("*.npz"):
            link(npz, struct_dir / npz.name); n_struct += 1
        for pa in predictions.glob("*/pre_affinity_*.npz"):
            link(pa, struct_dir / pa.name)
        for js in (processed / "records").glob("*.json"):
            link(js, rec_dir / js.name)
        for pk in (processed / "mols").glob("*.pkl"):
            link(pk, mol_dir / pk.name)
        # MSA filenames contain both record and chain IDs.
        for ms in (processed / "msa").glob("*.npz"):
            link(ms, msa_dir / ms.name); n_msa += 1
        man = json.loads((processed / "manifest.json").read_text())
        recs = man["records"] if isinstance(man, dict) else man
        records.extend(recs)

    seen, uniq = set(), []
    for r in records:
        rid = str(r.get("id"))
        if rid not in seen:
            seen.add(rid); uniq.append(r)
    (args.out_dir / "manifest.json").write_text(json.dumps({"records": uniq}))

    yaml_link = args.out_dir / "yamls"
    if yaml_link.is_symlink() or yaml_link.exists():
        yaml_link.unlink()
    os.symlink(args.inputs_dir.resolve(), yaml_link)

    print(f"merged {len(results_dirs)} chunks -> {args.out_dir}")
    print(f"embeddings={n_emb} structures={n_struct} msa={n_msa} "
          f"manifest_records={len(uniq)}")


if __name__ == "__main__":
    main()
