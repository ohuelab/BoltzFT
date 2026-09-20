#!/usr/bin/env python3
"""Build per-compound Boltz YAML inputs for an MF-PCBA target."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import yaml

LIGAND_CHAIN = "B"


def protein_seq_from_msa(msa_csv: Path) -> str:
    with open(msa_csv) as f:
        reader = csv.DictReader(f)
        first = next(reader)
    seq = first["sequence"]
    if "-" in seq or not seq:
        raise SystemExit(f"unexpected gapped/empty query row in {msa_csv}: {seq[:40]!r}")
    return seq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference-csv", type=Path, required=True,
                   help="inference_compounds.csv from select_subsets.py")
    p.add_argument("--target", required=True)
    p.add_argument("--msa-csv", type=Path, required=True)
    p.add_argument("--out-yaml-dir", type=Path, required=True)
    p.add_argument("--protein-seq", default=None,
                   help="override; default = MSA query row")
    p.add_argument("--smiles-col", default="neut-smiles")
    p.add_argument("--id-col", default="CID")
    args = p.parse_args()

    if not args.msa_csv.exists():
        raise SystemExit(f"MSA CSV not found: {args.msa_csv}")
    seq = args.protein_seq or protein_seq_from_msa(args.msa_csv)

    df = pd.read_csv(args.inference_csv)

    args.out_yaml_dir.mkdir(parents=True, exist_ok=True)
    msa_path = str(args.msa_csv.resolve())
    written = 0
    for _, row in df.iterrows():
        cid = int(row[args.id_col])
        smiles = str(row[args.smiles_col])
        complex_id = f"{args.target}_{cid}"
        doc = {
            "version": 1,
            "sequences": [
                {"protein": {"id": ["A"], "sequence": seq, "msa": msa_path}},
                {"ligand": {"id": [LIGAND_CHAIN], "smiles": smiles}},
            ],
            "properties": [{"affinity": {"binder": LIGAND_CHAIN}}],
        }
        (args.out_yaml_dir / f"{complex_id}.yaml").write_text(
            yaml.safe_dump(doc, sort_keys=False)
        )
        written += 1

    print(f"target={args.target} protein_len={len(seq)} msa={msa_path}")
    print(f"wrote {written} yaml -> {args.out_yaml_dir}")


if __name__ == "__main__":
    main()
