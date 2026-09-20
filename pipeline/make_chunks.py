#!/usr/bin/env python3
"""Group YAML inputs into chunks and write their input/output directory index."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--inputs-dir", type=Path, required=True, help="runs/<target>/inputs (yaml files)")
    p.add_argument("--chunks-root", type=Path, required=True, help="runs/<target>/inputs_chunks")
    p.add_argument("--outputs-root", type=Path, required=True, help="runs/<target>/outputs_chunks")
    p.add_argument("--chunk-size", type=int, default=350)
    p.add_argument("--index-file", type=Path, required=True, help="TSV index written for this target")
    args = p.parse_args()

    yamls = sorted(args.inputs_dir.glob("*.yaml"))
    if not yamls:
        raise SystemExit(f"no yaml files in {args.inputs_dir}")
    n_chunks = math.ceil(len(yamls) / args.chunk_size)

    args.chunks_root.mkdir(parents=True, exist_ok=True)
    lines = []
    for c in range(n_chunks):
        cdir = args.chunks_root / f"chunk_{c:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        for y in yamls[c * args.chunk_size:(c + 1) * args.chunk_size]:
            link = cdir / y.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(y.resolve())
        out_dir = args.outputs_root / f"chunk_{c:03d}"
        idx = c + 1  # One-based chunk index
        lines.append(f"{idx}\t{args.target}\t{cdir.resolve()}\t{out_dir.resolve()}")

    args.index_file.parent.mkdir(parents=True, exist_ok=True)
    args.index_file.write_text("\n".join(lines) + "\n")
    print(f"target={args.target} yamls={len(yamls)} chunks={n_chunks} "
          f"(chunk_size={args.chunk_size}) -> {args.chunks_root}")
    print(f"index now has {len(lines)} rows: {args.index_file}")


if __name__ == "__main__":
    main()
