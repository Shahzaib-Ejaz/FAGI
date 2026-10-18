import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import cicids_folds as C
from fagi.model.dense import DenseFAGI
from fagi.privacy.dp_sgd import compute_epsilon, steps_for


def fake_cicids(path, n=600, seed=0):
    rng = np.random.RandomState(seed)
    cols = [" Destination Port", " Flow Duration", "Flow Bytes/s", " Total Fwd Packets", " Label"]
    labels = np.array(["BENIGN"] * (n // 2) + ["DDoS"] * (n // 4) + ["PortScan"] * (n - n // 2 - n // 4))
    df = pd.DataFrame({c: rng.rand(n) * 100 + (labels != "BENIGN") * 30 for c in cols[:-1]})
    df[" Label"] = labels
    df.to_csv(path, index=False, encoding="utf-8")


@pytest.fixture(scope="module")
def folds(tmp_path_factory):
    d = tmp_path_factory.mktemp("cic")
    fake_cicids(d / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    return C.CICIDSFolds(str(d)), C.CICIDSFolds(str(d), protocol="temporal")


class TestPrivacyUnit:
    def test_private_pool_test_disjoint(self, folds):
        for F in folds:
            d = F.fold(0)
            assert not set(d["priv"]) & set(d["pool"])
            assert not set(d["te"]) & (set(d["priv"]) | set(d["pool"]))
            assert set(d["priv"]) | set(d["pool"]) == set(d["tr"])

    def test_private_graphs_contain_one_private_record(self, folds):
        F = folds[0]
        d = F.fold(0)
        assert np.allclose(d["Gp"][:, 0, :], d["Zp"])
        _, nbr = d["eb"].nn.kneighbors(d["Zp"], n_neighbors=C.CONFIG["k"])
        assert np.allclose(d["Gp"][:, 1:, :], d["Zr"][nbr])
        assert d["eb"].pool is d["Zr"]

    def test_preprocessor_fit_on_pool_only(self, folds):
        d = folds[0].fold(0)
        Z = d["pp"].transform(folds[0].X[d["pool"]])
        assert np.allclose(Z.mean(axis=0), 0, atol=1e-4)

    def test_temporal_split_respects_file_order(self, folds):
        tr, te = folds[1].splits[0]
        assert tr.max() < te.min()


class TestAttackHelpers:
    def test_knn_vote_and_tie(self, folds):
        F = folds[0]
        d = {"yr": np.array([0, 0, 1, 1, 1])}
        idx = np.array([[2, 3, 4, 0], [0, 1, 2, 3], [3, 0, 1, 2]])
        assert F.knn_predict(d, idx).tolist() == [1, 0, 1]
        assert F.knn_predict(d, np.array([[0, 1, 2, 3]]))[0] == 0

    def test_mimicry_respects_budget(self, folds):
        F = folds[0]
        d = F.fold(0)
        Z = d["Zte"][:20]
        seen = []
        F.mimicry_detected(lambda c: (seen.append(c - Z), np.ones(len(c), dtype=int))[1], d, Z, 0.1)
        assert len(seen) == C.CONFIG["mimicry_targets"]
        assert max(np.abs(s).max() for s in seen) <= 0.1 + 1e-6

    def test_adaptive_pgd_budget_and_best_iterate(self, folds):
        F = folds[0]
        d = F.fold(0)
        torch.manual_seed(0)
        model = DenseFAGI(d["Gp"].shape[2], gnnguard_threshold=None, readout="center").eval()
        Z = d["Zte"][:30]
        y = np.ones(len(Z), dtype=np.int64)
        za = F.adaptive_pgd(model, d, Z, y, 0.2, 5, 2, seed=1)
        assert np.abs(za - Z).max() <= 0.2 + 1e-5
        pred = F.predictor("gnn", model, d)(za)
        base = F.predictor("gnn", model, d)(Z)
        assert (pred == 0).sum() >= (base == 0).sum()


def test_matched_sigma_reaches_target():
    from exp4_federated import matched_sigma
    from synthetic_common import SYN_CONFIG
    target = 5.0
    s = matched_sigma(target, 2400, 30)
    q = SYN_CONFIG["batch_size"] / 2400
    assert compute_epsilon(s, q, steps_for(30, q), SYN_CONFIG["dp_delta"]) == pytest.approx(target, rel=1e-4)
