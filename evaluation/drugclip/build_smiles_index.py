"""Collect the unique SMILES over the 8 MF-PCBA targets into one index."""
import glob, os
import pandas as pd

here = os.environ["COMPARATOR_WORKDIR"]
rows = {}
for p in sorted(glob.glob(f"{here}/tables/*.csv")):
    df = pd.read_csv(p, usecols=["smiles"])
    for s in df.smiles:
        if s not in rows:
            rows[s] = len(rows)
idx = pd.DataFrame({"smi_id": list(rows.values()), "smiles": list(rows.keys())})
idx = idx.sort_values("smi_id")
idx.to_csv(f"{here}/smiles_index.csv", index=False)
print("unique smiles", len(idx))
