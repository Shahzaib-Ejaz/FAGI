"""UNSW-NB15 loader (the official training and testing partitions, combined and re-split by cross-validation).

Expected files (from the UNSW-NB15 "Training and Testing Sets" folder):
    UNSW_NB15_training-set.csv   175,341 rows
    UNSW_NB15_testing-set.csv     82,332 rows
Both have 45 columns: id, 39 numeric flow features, the categorical proto, service and state, attack_cat and label.

Cleaning mirrors data/cicids2017.py: drop rows with non-finite values, drop exact duplicates (features and label),
drop constant features. The three categorical features are dropped: an L-inf perturbation of a one-hot or ordinal
code has no traffic meaning, and the numeric features already encode most of what they carry (ttl, window and
port counters). attack_cat becomes the label, with "Normal" renamed BENIGN so the CICIDS2017 sampler applies.
"""
import hashlib
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data.cicids2017 import stratified_sample

FILES = {"UNSW_NB15_training-set.csv": 175341, "UNSW_NB15_testing-set.csv": 82332}
SHA256 = {"UNSW_NB15_training-set.csv": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
          "UNSW_NB15_testing-set.csv": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"}
DROP = ["id", "proto", "service", "state", "label"]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clean_flows(data_path: str) -> Tuple[pd.DataFrame, Dict]:
    frames, meta = [], {"files": {}}
    for name, rows in FILES.items():
        p = os.path.join(data_path, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found; download the UNSW-NB15 training and testing sets (see README)")
        df = pd.read_csv(p, low_memory=False)
        if len(df) != rows:
            raise ValueError(f"{name}: {len(df)} rows, expected {rows}")
        meta["files"][name] = {"rows": int(len(df)), "sha256": _sha256(p), "expected_sha256": SHA256[name]}
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Label"] = df["attack_cat"].astype(str).str.strip().replace({"Normal": "BENIGN"})
    df = df.drop(columns=DROP + ["attack_cat"])
    feats = [c for c in df.columns if c != "Label"]
    meta["raw_rows"] = int(len(df))
    meta["raw_label_counts"] = {k: int(v) for k, v in df["Label"].value_counts().items()}
    vals = df[feats].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(vals).all(axis=1)
    meta["dropped_nonfinite"] = int(bad.sum())
    df = df.loc[~bad].reset_index(drop=True)
    dup = df.duplicated(subset=feats + ["Label"])
    meta["dropped_duplicates"] = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)
    meta["rows_with_conflicting_labels"] = int(df.duplicated(subset=feats, keep=False).sum())
    const = [c for c in feats if df[c].nunique() <= 1]
    meta["dropped_constant_features"] = const
    meta["dropped_categorical_features"] = ["proto", "service", "state"]
    feats = [c for c in feats if c not in const]
    df = df[feats + ["Label"]].copy()
    df[feats] = df[feats].astype(np.float32)
    meta["clean_rows"] = int(len(df))
    meta["clean_label_counts"] = {k: int(v) for k, v in df["Label"].value_counts().items()}
    meta["features"] = feats
    return df, meta


def load_unsw_nb15(data_path: str = "data/unsw_nb15/", per_class_cap: int = 3000, seed: int = 42):
    df, meta = load_clean_flows(data_path)
    sample = stratified_sample(df, per_class_cap, seed)
    feats = meta["features"]
    X = sample[feats].to_numpy(dtype=np.float32)
    labels = sample["Label"].to_numpy()
    y = (labels != "BENIGN").astype(np.int64)
    meta = dict(meta, sample_per_class_cap=per_class_cap, sample_seed=seed, sample_rows=int(len(sample)),
                sample_label_counts={k: int(v) for k, v in pd.Series(labels).value_counts().items()})
    return X, y, labels, feats, meta


if __name__ == "__main__":
    X, y, labels, feats, meta = load_unsw_nb15()
    print(json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=1))
    print(len(feats), "features")
