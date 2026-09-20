"""Score self-contained ``affinity_cropped_v1`` caches with an affinity head."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch import nn

from boltz.data.crop_embeddings import load_affinity_cropped_inputs
from boltz.model.modules.affinity import AffinityModule


class CachedAffinityHeads(nn.Module):
    """Load affinity heads from a Boltz checkpoint."""

    def __init__(self, checkpoint: str, use_kernels: bool) -> None:
        super().__init__()
        saved = torch.load(
            checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
        hparams = saved["hyper_parameters"]
        self.affinity_ensemble = bool(hparams["affinity_ensemble"])
        self.affinity_mw_correction = bool(hparams.get("affinity_mw_correction", False))
        self.use_kernels = use_kernels
        names = (
            ["affinity_module1", "affinity_module2"]
            if self.affinity_ensemble
            else ["affinity_module"]
        )
        for name in names:
            args_key = f"{name.replace('affinity_module', 'affinity_model_args')}"
            args = hparams[args_key]
            module = AffinityModule(hparams["token_s"], hparams["token_z"], **args)
            prefix = f"{name}."
            state = {
                key[len(prefix):]: value
                for key, value in saved["state_dict"].items()
                if key.startswith(prefix)
            }
            module.load_state_dict(state, strict=True)
            setattr(self, name, module)


CACHE_PREFIX = "affinity_embeddings_"
CACHE_INDEX_NAME = "cache_index.json"


def cache_index(cache_dir: Path) -> dict[str, str]:
    """Map record id -> cache path, memoized in ``<cache_dir>/cache_index.json``."""
    index_path = cache_dir / CACHE_INDEX_NAME
    if index_path.is_file():
        return json.loads(index_path.read_text())

    index: dict[str, str] = {}
    for path in cache_dir.rglob(f"{CACHE_PREFIX}*.npz"):
        record_id = path.stem[len(CACHE_PREFIX):]
        if record_id in index:
            raise ValueError(
                f"Duplicate cache for {record_id}: {index[record_id]} and {path}"
            )
        index[record_id] = str(path)

    # Atomic publish: several scoring jobs can race to build this.
    tmp_path = index_path.with_suffix(f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(index))
    os.replace(tmp_path, index_path)
    return index


def _device_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def score_cache(
    model: CachedAffinityHeads, path: Path, device: torch.device
) -> dict[str, float]:
    cached = _device_batch(load_affinity_cropped_inputs(path), device)
    feats = {
        key: cached[key]
        for key in ("token_to_rep_atom", "token_pad_mask", "mol_type", "affinity_token_mask")
    }
    kwargs = dict(
        s_inputs=cached["s_inputs"],
        z=cached["z"],
        x_pred=cached["coords"],
        feats=feats,
        multiplicity=1,
        use_kernels=model.use_kernels,
    )
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        if model.affinity_ensemble:
            out1 = model.affinity_module1(**kwargs)
            out2 = model.affinity_module2(**kwargs)
            prob1 = torch.sigmoid(out1["affinity_logits_binary"])
            prob2 = torch.sigmoid(out2["affinity_logits_binary"])
            value1 = out1["affinity_pred_value"]
            value2 = out2["affinity_pred_value"]
            result = {
                "affinity_pred_value": ((value1 + value2) / 2).item(),
                "affinity_probability_binary": ((prob1 + prob2) / 2).item(),
                "affinity_pred_value1": value1.item(),
                "affinity_probability_binary1": prob1.item(),
                "affinity_pred_value2": value2.item(),
                "affinity_probability_binary2": prob2.item(),
            }
        else:
            out = model.affinity_module(**kwargs)
            result = {
                "affinity_pred_value": out["affinity_pred_value"].item(),
                "affinity_probability_binary": torch.sigmoid(
                    out["affinity_logits_binary"]
                ).item(),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--split-file", help="optional record ids, one per line")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-kernels", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = CachedAffinityHeads(
        args.pretrained, use_kernels=not args.no_kernels
    ).eval().to(device)

    cache_dir = Path(args.cache_dir)
    prefix = CACHE_PREFIX
    if args.split_file:
        ids = [
            line.strip()
            for line in Path(args.split_file).read_text().splitlines()
            if line.strip()
        ]
        paths = [cache_dir / f"{prefix}{record_id}.npz" for record_id in ids]
        if not all(path.is_file() for path in paths):
            # Find records across chunk subdirectories.
            index = {
                record_id: Path(value)
                for record_id, value in cache_index(cache_dir).items()
            }
            absent = [record_id for record_id in ids if record_id not in index]
            if absent:
                raise FileNotFoundError(
                    f"Missing {len(absent)} caches under {cache_dir}; first: {absent[0]}"
                )
            paths = [index[record_id] for record_id in ids]
    else:
        paths = sorted(Path(value) for value in cache_index(cache_dir).values())
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} caches; first: {missing[0]}")

    rows = []
    for idx, path in enumerate(paths, 1):
        record_id = path.stem[len(prefix):]
        rows.append({"sample_id": record_id, **score_cache(model, path, device)})
        if idx % 20 == 0 or idx == len(paths):
            print(f"Scored {idx}/{len(paths)}", flush=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} scores -> {out_path}")


if __name__ == "__main__":
    main()
