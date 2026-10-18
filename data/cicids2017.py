import glob
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

EXPECTED_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

DUPLICATE_COLUMNS = ["Fwd Header Length.1"]


def normalize_label(label: str) -> str:
    s = str(label).strip()
    for bad in ("ï¿½", "�", "\u0096", "–"):
        s = s.replace(bad, "-")
    return " ".join(s.split())


def _find_csvs(data_path: str) -> List[str]:
    found = sorted(glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True))
    if not found:
        raise FileNotFoundError(
            f"No CICIDS2017 CSV files under {data_path}. Extract MachineLearningCSV.zip there."
        )
    return found


def _read_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in DUPLICATE_COLUMNS if c in df.columns])
    df["Label"] = df["Label"].map(normalize_label)
    feats = [c for c in df.columns if c != "Label"]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    df["source_file"] = os.path.basename(path)
    return df


def load_clean_flows(data_path: str, cache_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    files = _find_csvs(data_path)
    key = hashlib.sha1("|".join(f"{os.path.basename(f)}:{os.path.getsize(f)}" for f in files).encode()).hexdigest()[:12]
    cache_dir = cache_dir or os.path.join(data_path, ".cache")
    cache_pq = os.path.join(cache_dir, f"clean_{key}.pkl")
    cache_meta = os.path.join(cache_dir, f"clean_{key}.json")
    if os.path.exists(cache_pq) and os.path.exists(cache_meta):
        with open(cache_meta) as f:
            return pd.read_pickle(cache_pq), json.load(f)

    frames = [_read_one(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    del frames
    feats = [c for c in df.columns if c not in ("Label", "source_file")]
    meta = {"files": [os.path.basename(f) for f in files], "raw_rows": int(len(df)),
            "raw_label_counts": {k: int(v) for k, v in df["Label"].value_counts().items()}}

    vals = df[feats].to_numpy()
    bad = ~np.isfinite(vals).all(axis=1)
    meta["dropped_nonfinite"] = int(bad.sum())
    df = df.loc[~bad].reset_index(drop=True)

    dup = df.duplicated(subset=feats + ["Label"])
    meta["dropped_duplicates"] = int(dup.sum())
    df = df.loc[~dup].reset_index(drop=True)

    conflict = df.duplicated(subset=feats, keep=False)
    meta["rows_with_conflicting_labels"] = int(conflict.sum())

    const = [c for c in feats if df[c].nunique() <= 1]
    meta["dropped_constant_features"] = const
    df = df.drop(columns=const)
    df[[c for c in feats if c not in const]] = df[[c for c in feats if c not in const]].astype(np.float32)

    meta["clean_rows"] = int(len(df))
    meta["clean_label_counts"] = {k: int(v) for k, v in df["Label"].value_counts().items()}
    meta["features"] = [c for c in feats if c not in const]

    os.makedirs(cache_dir, exist_ok=True)
    df.to_pickle(cache_pq)
    with open(cache_meta, "w") as f:
        json.dump(meta, f, indent=2)
    return df, meta


def stratified_sample(df: pd.DataFrame, per_class_cap: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    parts = []
    attack_total = 0
    for label in sorted(df["Label"].unique()):
        if label == "BENIGN":
            continue
        idx = np.flatnonzero(df["Label"].to_numpy() == label)
        take = idx if len(idx) <= per_class_cap else rng.choice(idx, per_class_cap, replace=False)
        parts.append(take)
        attack_total += len(take)
    benign = np.flatnonzero(df["Label"].to_numpy() == "BENIGN")
    parts.append(rng.choice(benign, min(attack_total, len(benign)), replace=False))
    sel = np.sort(np.concatenate(parts))
    return df.iloc[sel].reset_index(drop=True)


def load_cicids2017(
    data_path: str = "data/cicids2017/",
    per_class_cap: int = 1000,
    seed: int = 42,
    return_source: bool = False,
):
    df, meta = load_clean_flows(data_path)
    sample = stratified_sample(df, per_class_cap, seed)
    feats = meta["features"]
    X = sample[feats].to_numpy(dtype=np.float32)
    labels = sample["Label"].to_numpy()
    y = (labels != "BENIGN").astype(np.int64)
    meta = dict(meta)
    meta["sample_per_class_cap"] = per_class_cap
    meta["sample_seed"] = seed
    meta["sample_rows"] = int(len(sample))
    meta["sample_label_counts"] = {k: int(v) for k, v in pd.Series(labels).value_counts().items()}
    if return_source:
        return X, y, labels, feats, meta, sample["source_file"].to_numpy()
    return X, y, labels, feats, meta


class FlowPreprocessor:
    def fit(self, X: np.ndarray) -> "FlowPreprocessor":
        Z = np.sign(X) * np.log1p(np.abs(X))
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = np.sign(X) * np.log1p(np.abs(X))
        return ((Z - self.mean_) / self.std_).astype(np.float32)


class EgoGraphBuilder:
    def __init__(self, k: int = 8, k_inner: int = 3):
        self.k = k
        self.k_inner = k_inner

    def fit(self, pool: np.ndarray) -> "EgoGraphBuilder":
        self.pool = pool
        self.nn = NearestNeighbors(n_neighbors=self.k + 1, algorithm="brute", n_jobs=-1).fit(pool)
        return self

    def neighbours(self, Q: np.ndarray, self_index: Optional[np.ndarray] = None) -> np.ndarray:
        _, idx = self.nn.kneighbors(Q, n_neighbors=self.k + 1)
        if self_index is None:
            return idx[:, : self.k]
        out = np.empty((len(Q), self.k), dtype=np.int64)
        for r in range(len(Q)):
            row = idx[r][idx[r] != self_index[r]]
            out[r] = row[: self.k]
        return out

    def adjacency(self, nodes: np.ndarray) -> np.ndarray:
        n = nodes.shape[1]
        d = ((nodes[:, :, None, :] - nodes[:, None, :, :]) ** 2).sum(-1)
        idx = np.arange(n)
        d[:, idx, idx] = np.inf
        A = np.zeros((len(nodes), n, n), dtype=np.float32)
        A[:, 0, 1:] = 1.0
        A[:, 1:, 0] = 1.0
        nearest = np.argsort(d, axis=2)[:, :, : self.k_inner]
        b = np.arange(len(nodes))[:, None, None]
        A[b, idx[None, :, None], nearest] = 1.0
        A = np.maximum(A, A.transpose(0, 2, 1))
        A[:, idx, idx] = 0.0
        return A

    def build(self, Q: np.ndarray, self_index: Optional[np.ndarray] = None,
              nbr_idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if nbr_idx is None:
            nbr_idx = self.neighbours(Q, self_index)
        nodes = np.concatenate([Q[:, None, :], self.pool[nbr_idx]], axis=1).astype(np.float32)
        return nodes, self.adjacency(nodes), nbr_idx


def center_mask(n_graphs: int, n_nodes: int) -> np.ndarray:
    m = np.zeros((n_graphs, n_nodes), dtype=np.float32)
    m[:, 0] = 1.0
    return m
