# Adapted for MF-PCBA inputs from DrugCLIP.
# Copyright (c) 2025 AIR, Tsinghua University; portions (c) 2022 DP Technology.
# SPDX-License-Identifier: Apache-2.0
# See ../../licenses/DrugCLIP.txt.
"""DrugCLIP model loading and featurisation."""
import os
import sys

import numpy as np
import torch
from scipy.spatial import distance_matrix

DRUGCLIP = os.environ["DRUGCLIP_ROOT"]
CKPT = os.environ["DRUGCLIP_CHECKPOINT"]
if DRUGCLIP not in sys.path:
    sys.path.insert(0, DRUGCLIP)


def load_dicts():
    from unicore.data import Dictionary

    mol_dict = Dictionary.load(os.path.join(DRUGCLIP, "data/dict_mol.txt"))
    pkt_dict = Dictionary.load(os.path.join(DRUGCLIP, "data/dict_pkt.txt"))
    mol_dict.add_symbol("[MASK]", is_special=True)
    pkt_dict.add_symbol("[MASK]", is_special=True)
    return mol_dict, pkt_dict


def build_model(ckpt_path=CKPT, device="cuda"):
    import unimol  # noqa: F401  (registers arch)
    from unimol.models.drugclip import BindingAffinityModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    args.arch = "drugclip"
    mol_dict, pkt_dict = load_dicts()
    model = BindingAffinityModel(args, mol_dict, pkt_dict)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    return model, mol_dict, pkt_dict


def pocket_element(atom_name):
    """unimol AffinityPocketDataset.pocket_atom"""
    if atom_name[0] in "0123456789":
        return atom_name[1]
    return atom_name[0]


def _tokens_coords(atoms, coords, dictionary, max_seq_len=512):
    """atoms: list[str] element symbols (H already removed). coords: (n,3)."""
    coords = np.asarray(coords, dtype=np.float32)
    coords = coords - coords.mean(axis=0)
    idx = np.array([dictionary.index(a) for a in atoms], dtype=np.int64)
    idx = idx[: max_seq_len - 1]  # TokenizeDataset asserts len < max_seq_len
    coords = coords[: len(idx)]
    tokens = np.concatenate([[dictionary.bos()], idx, [dictionary.eos()]])
    n = len(tokens)
    dist = np.zeros((n, n), dtype=np.float32)
    dist[1:-1, 1:-1] = distance_matrix(coords, coords).astype(np.float32)
    return tokens, dist


def featurize_mol(atoms, coords, mol_dict):
    keep = np.array([a != "H" for a in atoms])
    atoms = [a for a, k in zip(atoms, keep) if k]
    coords = np.asarray(coords, dtype=np.float32)[keep]
    return _tokens_coords(atoms, coords, mol_dict)


def featurize_pocket(atom_names, coords, pkt_dict, seed=1, max_pocket_atoms=511):
    elems = np.array([pocket_element(a) for a in atom_names])
    coords = np.asarray(coords, dtype=np.float32)
    keep = elems != "H"
    elems, coords = elems[keep], coords[keep]
    if max_pocket_atoms and len(elems) > max_pocket_atoms:
        # unimol CroppingPocketDataset (seeded softmax-of-inverse-distance sampling)
        from unimol.data import data_utils

        with data_utils.numpy_seed(seed, None, 0):
            distance = np.linalg.norm(coords - coords.mean(axis=0), axis=1)

            def softmax(x):
                x = x - np.max(x)
                return np.exp(x) / np.sum(np.exp(x))

            distance = distance + 1
            weight = softmax(np.reciprocal(distance))
            index = np.random.choice(
                len(elems), max_pocket_atoms, replace=False, p=weight
            )
            elems, coords = elems[index], coords[index]
    return _tokens_coords(list(elems), coords, pkt_dict)


def collate(items, pad_idx):
    """items: list of (tokens, dist). Returns padded tensors."""
    n = max(len(t) for t, _ in items)
    b = len(items)
    tok = np.full((b, n), pad_idx, dtype=np.int64)
    dist = np.zeros((b, n, n), dtype=np.float32)
    for i, (t, d) in enumerate(items):
        tok[i, : len(t)] = t
        dist[i, : len(t), : len(t)] = d
    return torch.from_numpy(tok), torch.from_numpy(dist)


@torch.no_grad()
def encode(model, tok, dist, num_types, which, device="cuda", fp16=True):
    """Returns (encoder CLS rep [B,512], projected+normalised emb [B,128])."""
    sub = model.mol_model if which == "mol" else model.pocket_model
    proj = model.mol_project if which == "mol" else model.pocket_project
    tok = tok.to(device)
    dist = dist.to(device)
    edge = tok.view(tok.size(0), -1, 1) * num_types + tok.view(tok.size(0), 1, -1)
    padding_mask = tok.eq(sub.padding_idx)
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        x = sub.embed_tokens(tok)
        gbf = sub.gbf(dist, edge)
        bias = sub.gbf_proj(gbf)
        n_node = dist.size(-1)
        bias = bias.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)
        out = sub.encoder(x, padding_mask=padding_mask, attn_mask=bias)
        rep = out[0][:, 0, :]
        emb = proj(rep)
    emb = emb.float()
    emb = emb / emb.norm(dim=1, keepdim=True)
    return rep.float().cpu(), emb.cpu()
