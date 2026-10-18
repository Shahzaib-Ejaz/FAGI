import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.cicids2017 import (EgoGraphBuilder, FlowPreprocessor, load_clean_flows, normalize_label,
                             stratified_sample)


def write_fake_csv(path, n=200, seed=0):
    rng = np.random.RandomState(seed)
    cols = [" Destination Port", " Flow Duration", "Flow Bytes/s", " Fwd Header Length", " Fwd Header Length.1", " Bwd PSH Flags", " Label"]
    df = pd.DataFrame({c: rng.rand(n) * 100 for c in cols[:-1]})
    df[" Bwd PSH Flags"] = 0
    labels = np.array(["BENIGN"] * (n // 2) + ["Web Attack � XSS"] * (n // 4) + ["DDoS"] * (n - n // 2 - n // 4))
    df[" Label"] = labels
    df.loc[0, "Flow Bytes/s"] = np.inf
    df.loc[1, " Flow Duration"] = np.nan
    df = pd.concat([df, df.iloc[[5, 6]]], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8")


@pytest.fixture
def fake_dir(tmp_path):
    write_fake_csv(tmp_path / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    return str(tmp_path)


class TestLoader:
    def test_label_normalization(self):
        assert normalize_label("Web Attack ï¿½ Brute Force") == "Web Attack - Brute Force"
        assert normalize_label(" Web Attack � XSS ") == "Web Attack - XSS"

    def test_cleaning(self, fake_dir):
        df, meta = load_clean_flows(fake_dir, cache_dir=os.path.join(fake_dir, "c"))
        assert meta["raw_rows"] == 202
        assert meta["dropped_nonfinite"] == 2
        assert meta["dropped_duplicates"] == 2
        assert "Bwd PSH Flags" in meta["dropped_constant_features"]
        assert "Fwd Header Length.1" not in df.columns
        assert "Destination Port" in meta["features"]
        assert set(df["Label"]) <= {"BENIGN", "DDoS", "Web Attack - XSS"}
        assert np.isfinite(df[meta["features"]].to_numpy()).all()

    def test_cache_roundtrip(self, fake_dir):
        c = os.path.join(fake_dir, "c")
        a, ma = load_clean_flows(fake_dir, cache_dir=c)
        b, mb = load_clean_flows(fake_dir, cache_dir=c)
        assert ma == mb and a.equals(b)

    def test_stratified_sample_balanced(self, fake_dir):
        df, _ = load_clean_flows(fake_dir, cache_dir=os.path.join(fake_dir, "c"))
        s = stratified_sample(df, per_class_cap=20, seed=0)
        vc = s["Label"].value_counts()
        assert vc["DDoS"] == 20 and vc["Web Attack - XSS"] == 20 and vc["BENIGN"] == 40


class TestEgoGraphs:
    def test_training_graphs_exclude_self(self):
        rng = np.random.RandomState(0)
        Z = rng.randn(50, 4).astype(np.float32)
        eb = EgoGraphBuilder(k=5, k_inner=2).fit(Z)
        G, A, nbr = eb.build(Z, self_index=np.arange(50))
        assert G.shape == (50, 6, 4) and A.shape == (50, 6, 6)
        assert not np.any(nbr == np.arange(50)[:, None])
        assert np.allclose(G[:, 0], Z)

    def test_adjacency_symmetric_star(self):
        Z = np.random.RandomState(1).randn(30, 3).astype(np.float32)
        eb = EgoGraphBuilder(k=4, k_inner=2).fit(Z)
        _, A, _ = eb.build(Z[:5])
        assert np.allclose(A, A.transpose(0, 2, 1))
        assert np.all(A[:, 0, 1:] == 1)
        assert np.all(A[:, np.arange(5), np.arange(5)] == 0)

    def test_preprocessor_fit_on_train_only(self):
        X = np.abs(np.random.RandomState(2).randn(100, 3)) * 1000
        pp = FlowPreprocessor().fit(X[:80])
        Z = pp.transform(X[:80])
        assert np.allclose(Z.mean(0), 0, atol=1e-4)
