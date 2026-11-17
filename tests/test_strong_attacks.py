import os
import sys

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import strong_attacks as sa
from data.cicids2017 import EgoGraphBuilder


class LinearTarget:
    """Score = sigmoid(w.z); stands in for a model so the optimizers can be tested without CICIDS2017."""
    kind = "flat"

    def __init__(self, w):
        self.w = torch.as_tensor(w, dtype=torch.float32)

    def score(self, Z):
        return torch.sigmoid(torch.as_tensor(Z) @ self.w).numpy()

    def score_and_grad(self, Z):
        x = torch.as_tensor(Z).clone().requires_grad_(True)
        m = x @ self.w
        (g,) = torch.autograd.grad(m.sum(), x)
        return torch.sigmoid(m).detach().numpy(), g.numpy()


def _data(n=300, m=40, d=12, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, d).astype(np.float32), rng.randn(m, d).astype(np.float32)


def test_retriever_matches_sklearn_neighbours():
    pool, Q = _data()
    r = sa.Retriever(pool, np.arange(len(pool)) % 2, 8, 3)
    idx, d2 = r.neighbours(torch.as_tensor(Q))
    dist, ref = NearestNeighbors(n_neighbors=8, algorithm="brute").fit(pool).kneighbors(Q)
    assert np.array_equal(idx.numpy(), ref)
    assert np.allclose(np.sqrt(d2.numpy()), dist, atol=1e-5)


def test_retriever_graph_matches_ego_graph_builder():
    pool, Q = _data()
    eb = EgoGraphBuilder(8, 3).fit(pool)
    G, A, nbr = eb.build(Q)
    r = sa.Retriever(pool, np.arange(len(pool)) % 2, 8, 3)
    nodes, adj = r.graph(torch.as_tensor(Q), torch.as_tensor(nbr))
    assert np.allclose(nodes.numpy(), G)
    assert np.array_equal(adj.numpy(), A)


def test_gradient_attack_reaches_the_linear_optimum_and_stays_in_budget():
    _, Z0 = _data()
    w = np.random.RandomState(1).randn(Z0.shape[1]).astype(np.float32)
    tg, eps = LinearTarget(w), 0.3
    best = sa.Best(Z0)
    sa.gradient_attack(tg, Z0, eps, best, steps=20, restarts=2, rng=np.random.RandomState(0))
    optimum = tg.score(Z0 - eps * np.sign(w))
    assert np.abs(best.z - Z0).max() <= eps + 1e-6
    assert np.allclose(best.s, tg.score(best.z), atol=1e-6)
    done = optimum <= sa.SUITE["score_floor"]
    assert np.allclose(best.s[~done], optimum[~done], atol=1e-4)
    assert (best.s[done] <= sa.SUITE["score_floor"] + 1e-6).all()


def test_random_search_never_raises_the_score_and_stays_in_budget():
    _, Z0 = _data()
    w = np.random.RandomState(2).randn(Z0.shape[1]).astype(np.float32)
    tg, eps = LinearTarget(w), 0.2
    best = sa.Best(Z0)
    best.offer(Z0, tg.score(Z0))
    start = best.s.copy()
    mimic = np.repeat(Z0[:, None, :], 4, axis=1) + 1.0
    sa.random_search(tg, Z0, eps, best, 60, np.random.RandomState(0), mimic)
    assert (best.s <= start + 1e-9).all() and (best.s < start - 1e-4).mean() > 0.9
    assert np.abs(best.z - Z0).max() <= eps + 1e-6
    assert np.allclose(best.s, tg.score(best.z), atol=1e-6)


def test_by_class_returns_nearest_per_label_and_overall():
    pool, Q = _data()
    y = np.arange(len(pool)) % 2
    r = sa.Retriever(pool, y, 8, 3)
    ib, im, ik = r.by_class(torch.as_tensor(Q), 5, 4)
    for lab, got, k in ((0, ib, 5), (1, im, 4)):
        sub = np.flatnonzero(y == lab)
        _, ref = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(pool[sub]).kneighbors(Q)
        assert np.array_equal(got.numpy(), sub[ref])
    _, ref = NearestNeighbors(n_neighbors=8, algorithm="brute").fit(pool).kneighbors(Q)
    assert np.array_equal(ik.numpy(), ref)


def test_best_keeps_the_lowest_score_per_row():
    Z = np.zeros((3, 2), dtype=np.float32)
    b = sa.Best(Z)
    b.offer(Z + 1, np.array([0.9, 0.2, 0.5]))
    b.offer(Z + 2, np.array([0.1, 0.4, 0.5]))
    assert np.allclose(b.s, [0.1, 0.2, 0.5])
    assert np.allclose(b.z[:, 0], [2, 1, 1])
