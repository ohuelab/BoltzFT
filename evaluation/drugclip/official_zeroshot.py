# Adapted for MF-PCBA inputs from DrugCLIP.
# Copyright (c) 2025 AIR, Tsinghua University; portions (c) 2022 DP Technology.
# SPDX-License-Identifier: Apache-2.0
# See ../../licenses/DrugCLIP.txt.
"""Encode molecules and pockets using the DrugCLIP dataset and inference routines."""
import argparse, glob, os, pickle, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lmdb, numpy as np, pandas as pd, torch

import dc_model as D
from verify_tables import TARGETS

here = os.environ["COMPARATOR_WORKDIR"]


def write_lmdb(records, path):
    if os.path.exists(path):
        os.remove(path)
    env = lmdb.open(path, subdir=False, lock=False, readahead=False, meminit=False,
                    max_readers=64, map_size=1024 ** 4)
    with env.begin(write=True) as txn:
        for i, d in enumerate(records):
            txn.put(str(i).encode("ascii"), pickle.dumps(d))
    env.close()


def build_lmdbs(out_dir):
    """mols.lmdb over the unique SMILES, pockets.lmdb over the 8 targets."""
    os.makedirs(out_dir, exist_ok=True)
    mol_path = f"{out_dir}/mols.lmdb"
    if not os.path.exists(mol_path):
        idx = pd.read_csv(f"{here}/smiles_index.csv")
        smi_by_id = dict(zip(idx.smi_id, idx.smiles))
        records = []
        for p in sorted(glob.glob(f"{here}/confs/shard_[0-9][0-9][0-9][0-9].npz")):
            z = np.load(p, allow_pickle=True)
            for sid, atoms, coords in zip(z["smi_id"], z["atoms"], z["coords"]):
                records.append({
                    "atoms": list(atoms),
                    "coordinates": [np.asarray(c, dtype=np.float32) for c in coords],
                    "smi": smi_by_id[int(sid)],
                    "label": 0,
                    "smi_id": int(sid),
                })
        print(f"writing {len(records)} molecules -> {mol_path}", flush=True)
        write_lmdb(records, mol_path)
    pkt_path = f"{out_dir}/pockets.lmdb"
    if not os.path.exists(pkt_path):
        recs = []
        for t in TARGETS:
            z = np.load(f"{here}/pockets/{t}.npz")
            recs.append({"pocket_atoms": list(z["pocket_atoms"]),
                         "pocket_coordinates": [c for c in z["pocket_coordinates"]],
                         "pocket": t})
        write_lmdb(recs, pkt_path)
    return mol_path, pkt_path


def make_task():
    import unimol  # noqa: F401
    from unimol.tasks.drugclip import DrugCLIP
    from unicore.data import Dictionary
    args = types.SimpleNamespace(seed=1, max_seq_len=512, max_pocket_atoms=511,
                                 data=os.path.join(D.DRUGCLIP, "data"), dist_threshold=6.0)
    mol_d = Dictionary.load(os.path.join(D.DRUGCLIP, "data/dict_mol.txt"))
    pkt_d = Dictionary.load(os.path.join(D.DRUGCLIP, "data/dict_pkt.txt"))
    return DrugCLIP(args, mol_d, pkt_d)


# Adapted from DrugCLIP unimol/tasks/drugclip.py::test_pcba_target.
def official_mol_batch(model, sample):
    import unicore.utils
    sample = unicore.utils.move_to_cuda(sample)
    dist = sample["net_input"]["mol_src_distance"]
    et = sample["net_input"]["mol_src_edge_type"]
    st = sample["net_input"]["mol_src_tokens"]
    mol_padding_mask = st.eq(model.mol_model.padding_idx)
    mol_x = model.mol_model.embed_tokens(st)
    n_node = dist.size(-1)
    gbf_feature = model.mol_model.gbf(dist, et)
    gbf_result = model.mol_model.gbf_proj(gbf_feature)
    graph_attn_bias = gbf_result
    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
    mol_outputs = model.mol_model.encoder(
        mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias)
    mol_encoder_rep = mol_outputs[0][:, 0, :]
    mol_emb = model.mol_project(mol_encoder_rep)
    mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)
    return mol_emb.detach().cpu().numpy(), sample["smi_name"]


def official_pocket_batch(model, sample):
    import unicore.utils
    sample = unicore.utils.move_to_cuda(sample)
    dist = sample["net_input"]["pocket_src_distance"]
    et = sample["net_input"]["pocket_src_edge_type"]
    st = sample["net_input"]["pocket_src_tokens"]
    pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
    pocket_x = model.pocket_model.embed_tokens(st)
    n_node = dist.size(-1)
    gbf_feature = model.pocket_model.gbf(dist, et)
    gbf_result = model.pocket_model.gbf_proj(gbf_feature)
    graph_attn_bias = gbf_result
    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
    pocket_outputs = model.pocket_model.encoder(
        pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias)
    pocket_encoder_rep = pocket_outputs[0][:, 0, :]
    pocket_emb = model.pocket_project(pocket_encoder_rep)
    pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)
    return pocket_emb.detach().cpu().numpy(), sample["pocket_name"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb-dir", default=os.environ.get(
        "DC_LMDB", f"{here}/lmdb"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=f"{here}/results/official_zeroshot.npz")
    args = ap.parse_args()

    mol_path, pkt_path = build_lmdbs(args.lmdb_dir)
    task = make_task()
    model, mol_dict, pkt_dict = D.build_model()

    pds = task.load_pockets_dataset(pkt_path)
    ploader = torch.utils.data.DataLoader(pds, batch_size=8, collate_fn=pds.collater)
    pemb, pnames = [], []
    for s in ploader:
        e, n = official_pocket_batch(model, s)
        pemb.append(e); pnames.extend(n)
    pemb = np.concatenate(pemb)

    mds = task.load_mols_dataset(mol_path, "atoms", "coordinates")
    loader = torch.utils.data.DataLoader(mds, batch_size=args.batch_size,
                                         collate_fn=mds.collater, num_workers=4)
    memb, mnames = [], []
    import time
    t0 = time.time()
    for i, s in enumerate(loader):
        e, n = official_mol_batch(model, s)
        memb.append(e); mnames.extend(n)
        if i % 200 == 0:
            print(f"  batch {i} n={len(mnames)} {time.time()-t0:.0f}s", flush=True)
    memb = np.concatenate(memb)
    print(f"encoded {len(memb)} molecules in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, emb=memb.astype(np.float16),
                        smiles=np.array(mnames), pocket=np.array(pnames),
                        pocket_emb=pemb.astype(np.float32))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
