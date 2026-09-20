#!/usr/bin/env python3
"""Pool affinity-stage interface representations into 128-dimensional features."""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


STRING_DTYPES = {"complex_id": str, "target": str, "sample_id": str}


def pool_strips(
    z_rows: np.ndarray,
    z_cols: np.ndarray,
    lig_idx: np.ndarray,
    valid: np.ndarray,
    length: int,
) -> np.ndarray:
    """Average ligand-receptor and off-diagonal ligand-ligand pairs."""
    if z_rows.shape[:2] != (len(lig_idx), length):
        raise ValueError(f"bad z_rows shape {z_rows.shape}")
    if z_cols.shape[:2] != (length, len(lig_idx)):
        raise ValueError(f"bad z_cols shape {z_cols.shape}")
    if valid.shape != (length,):
        raise ValueError(f"bad token_pad_mask shape {valid.shape}")
    if not valid[lig_idx].all():
        raise ValueError("lig_idx contains padded tokens")

    ligand = np.zeros(length, dtype=bool)
    ligand[lig_idx] = True
    receptor = valid & ~ligand
    # Count each ligand-ligand edge once.
    lr = z_rows[:, receptor, :].reshape(-1, z_rows.shape[-1])
    rl = z_cols[receptor, :, :].reshape(-1, z_rows.shape[-1])
    ll = z_rows[:, ligand, :]
    offdiag = ~np.eye(len(lig_idx), dtype=bool)
    parts = (lr, rl, ll[offdiag])
    n_pairs = sum(len(part) for part in parts)
    if n_pairs == 0:
        raise ValueError("no affinity cross pairs")
    # Accumulate in float64 to reduce rounding error.
    pooled = sum(
        (part.sum(axis=0, dtype=np.float64) for part in parts),
        start=np.zeros(z_rows.shape[-1], dtype=np.float64),
    ) / n_pairs
    return pooled.astype(np.float32)


def pool_one(item: tuple[str, str]) -> tuple[str, np.ndarray | None, str, str]:
    cid, path_string = item
    try:
        from boltz.data.crop_embeddings import load_affinity_cropped_inputs
        loaded = load_affinity_cropped_inputs(path_string)
        fmt = "affinity_cropped_v1"
        z = loaded["z"][0].numpy()
        valid = loaded["token_pad_mask"][0].numpy().astype(bool, copy=False)
        ligand = loaded["affinity_token_mask"][0].numpy().astype(bool, copy=False) & valid
        lig_idx = np.flatnonzero(ligand).astype(np.int64)
        length = z.shape[0]
        z_rows = z[lig_idx, :, :]
        z_cols = z[:, lig_idx, :]

        return cid, pool_strips(z_rows, z_cols, lig_idx, valid, length), "", fmt
    except Exception as exc:
        return cid, None, f"{type(exc).__name__}: {exc}", ""


def write_vectors(path: Path, ids: list[str], vectors: dict[str, np.ndarray],
                  formats: dict[str, str]) -> None:
    """Atomically write ordered vectors; used for both partial and final state."""
    present = [cid for cid in ids if cid in vectors]
    graw = (
        np.vstack([vectors[cid] for cid in present])
        if present else np.empty((0, 128), dtype=np.float32)
    )
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            ids=np.asarray(present),
            graw=graw,
            formats=np.asarray([formats.get(cid, "") for cid in present]),
        )
    temp.replace(path)


def load_partial(path: Path, expected: list[str]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if not path.exists():
        return {}, {}
    with np.load(path, allow_pickle=True) as data:
        ids = [str(x) for x in data["ids"]]
        graw = data["graw"].astype(np.float32)
        saved_formats = (
            [str(x) for x in data["formats"]]
            if "formats" in data.files else [""] * len(ids)
        )
    if len(ids) != len(set(ids)) or graw.shape != (len(ids), 128):
        raise SystemExit(f"invalid partial checkpoint: {path}")
    unexpected = set(ids) - set(expected)
    if unexpected:
        raise SystemExit(f"partial checkpoint has {len(unexpected)} unexpected ids: {path}")
    return dict(zip(ids, graw)), dict(zip(ids, saved_formats))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ml-table", type=Path, required=True)
    ap.add_argument("--embeddings-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output NPZ with ids and graw")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=1000,
                    help="atomically save successful vectors every N newly processed files")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to replace existing output: {args.out}")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be positive")
    ml = pd.read_csv(args.ml_table, dtype=STRING_DTYPES)
    expected = ml["complex_id"].astype(str).tolist()
    if len(expected) != len(set(expected)):
        raise SystemExit("duplicate complex_id values in ml_table")

    paths: dict[str, Path] = {}
    duplicate_paths: list[str] = []
    for pattern, prefix in (
        ("**/affinity_embeddings_*.npz", "affinity_embeddings_"),
    ):
        for path in args.embeddings_root.glob(pattern):
            cid = path.stem.removeprefix(prefix)
            if cid in paths:
                duplicate_paths.append(cid)
            else:
                paths[cid] = path
    if duplicate_paths:
        raise SystemExit(f"duplicate embedding paths for {len(duplicate_paths)} ids")

    partial_path = args.out.with_name(args.out.stem + "_partial.npz")
    if args.overwrite and partial_path.exists():
        partial_path.unlink()
    vectors, formats = load_partial(partial_path, expected)
    if vectors:
        print(f"resuming from {partial_path}: {len(vectors)}/{len(expected)} vectors", flush=True)

    items = [(cid, str(paths[cid])) for cid in expected if cid in paths and cid not in vectors]
    missing = [cid for cid in expected if cid not in paths]
    errors: dict[str, str] = {cid: "missing embedding file" for cid in missing}
    if items:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done, (cid, vec, error, fmt) in enumerate(
                pool.map(pool_one, items, chunksize=8), 1
            ):
                if error:
                    errors[cid] = error
                else:
                    vectors[cid] = vec
                    formats[cid] = fmt
                if done % args.checkpoint_every == 0 or done == len(items):
                    write_vectors(partial_path, expected, vectors, formats)
                    print(
                        f"{done}/{len(items)} new files; total_ok={len(vectors)}/"
                        f"{len(expected)} errors={len(errors)}; checkpoint={partial_path}",
                        flush=True,
                    )

    if errors:
        error_path = args.out.with_name(args.out.stem + "_errors.csv")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(list(errors.items()), columns=["complex_id", "error"]).to_csv(
            error_path, index=False
        )
        raise SystemExit(
            f"{len(errors)} embeddings failed; see {error_path}"
        )
    write_vectors(args.out, expected, vectors, formats)
    if partial_path.exists():
        partial_path.unlink()
    args.out.with_name(args.out.stem + "_errors.csv").unlink(missing_ok=True)
    print(f"Wrote {len(expected)} features to {args.out}", flush=True)


if __name__ == "__main__":
    main()
