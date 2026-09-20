"""Digest ordered compound IDs, labels and split membership."""
import hashlib
COLS = ["complex_id", "is_binder", "train_40", "train_100", "train_300", "is_eval"]

def digest(df):
    h = hashlib.sha256()
    for c in COLS:
        v = df[c].values
        if v.dtype == bool or str(v.dtype).startswith("int"):
            h.update(v.astype("int64").tobytes())
        else:
            h.update("\n".join(map(str, v)).encode())
        h.update(c.encode())
    return h.hexdigest()

TARGETS = ["588689", "540297-493091", "434954-2097", "463203-2650",
           "493248-485317", "504329", "624273-588549", "1053173-743445"]
