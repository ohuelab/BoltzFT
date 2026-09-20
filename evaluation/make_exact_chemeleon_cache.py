#!/usr/bin/env python3
"""Compute CheMeleon embeddings from the results CSV's SMILES."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from chemeleon_fingerprint import CheMeleonFingerprint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-csv", type=Path, required=True)
    ap.add_argument("--ml-table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to replace existing output: {args.out}")

    results = pd.read_csv(args.results_csv, dtype={"CID": str})
    ml = pd.read_csv(args.ml_table, dtype={"complex_id": str, "ligand_id": str})
    if results["CID"].duplicated().any():
        raise SystemExit("duplicate CID values in results CSV")
    smiles_by_cid = results.set_index("CID")["neut-smiles"]
    ids = ml["complex_id"].astype(str).tolist()
    ligand_ids = ml["ligand_id"].astype(str).tolist()
    missing = [cid for cid in ligand_ids if cid not in smiles_by_cid.index]
    if missing:
        raise SystemExit(f"results CSV is missing {len(missing)} ligand ids")
    smiles = smiles_by_cid.loc[ligand_ids].tolist()

    model = CheMeleonFingerprint(device=args.device)
    chunks = []
    failed: list[dict[str, str]] = []
    for start in range(0, len(smiles), args.batch_size):
        stop = min(start + args.batch_size, len(smiles))
        try:
            chunks.append(model(smiles[start:stop]).astype(np.float32))
        except Exception as batch_exc:
            # Retry individually to identify the invalid molecules.
            rows = []
            for offset, smi in enumerate(smiles[start:stop], start):
                try:
                    rows.append(model([smi])[0].astype(np.float32))
                except Exception as exc:
                    failed.append({"complex_id": ids[offset], "smiles": str(smi),
                                   "error": f"{type(exc).__name__}: {exc}"})
                    rows.append(None)
            if failed:
                error_path = args.out.with_name(args.out.stem + "_errors.csv")
                error_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(failed).to_csv(error_path, index=False)
                raise RuntimeError(
                    f"CheMeleon failed for {len(failed)} molecules; see {error_path}"
                ) from batch_exc
            chunks.append(np.vstack(rows))
        print(f"{stop}/{len(smiles)}", flush=True)
    emb = np.vstack(chunks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp_out = args.out.with_name(args.out.name + f".tmp-{os.getpid()}")
    with temp_out.open("wb") as handle:
        np.savez_compressed(handle, ids=np.asarray(ids), emb=emb)
    temp_out.replace(args.out)
    args.out.with_name(args.out.stem + "_errors.csv").unlink(missing_ok=True)
    print(f"Wrote {len(ids)} features to {args.out}", flush=True)


if __name__ == "__main__":
    main()
