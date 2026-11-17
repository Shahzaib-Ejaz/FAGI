"""Score-minimizing attack suite, run from the cached CICIDS2017 models.

The attacks in cicids_folds.py stop at the first iterate that crosses a model's default decision boundary, so
re-scoring them at another threshold gives an upper bound on recall there. The attacks here minimize each
model's malicious score instead and keep the lowest-scoring example found for every flow. One run therefore
serves every threshold: a flow is detected at threshold t only if its lowest score stays above t.

For every (model, fold, seed) and budget the suite is, in order:

1. The stored attacks, re-scored: mimicry toward each of the 32 nearest benign pool flows, the model's own
   stored gradient examples, the two stored transfer sources, and (graph models) PGD on the fixed neighborhood
   followed by re-retrieval.
2. A white-box attack on the score:
     neural models   PGD on the logit margin z_1 - z_0 (no cross-entropy saturation), with one start at the
                     clean flow, one at the best example of stage 1 and further random starts; graph models
                     re-retrieve neighbors at every step.
     k-NN vote       PGD on a soft vote, the label average of the nearest pool flows weighted by
                     exp(-d^2 / T); the true vote is evaluated at every step.
     random forest   none (no gradients); stages 3 and 4 are its attacks.
3. Transfer of the best stage-2 examples of every other source model.
4. Score-based random search from the best example so far: random sign flips on a shrinking subset of
   features, alternating with copying a random feature subset from a mimicry candidate; a proposal is kept
   when the queried score falls. It uses scores only, so it also checks the gradient attacks for masking.

Optimization uses an exact float64 re-ranking of a float32 candidate search for neighbor retrieval. The final
score of every flow's best example is then computed once more through the deployed inference path
(EgoGraphBuilder with scikit-learn retrieval), and only that value is reported.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from cicids_folds import CACHE_DIR, CONFIG, FLAT_VARIANTS, GRAPH_VARIANTS, TRANSFER_SOURCES, CICIDSFolds, as_graph, inverse_flow
from common import T, adversarial_inputs
from data.cicids2017 import center_mask

SUITE = {
    "version": 1,
    "graph_steps": 25,
    "graph_restarts": 2,
    "flat_steps": 100,
    "flat_restarts": 4,
    "knn_steps": 25,
    "knn_restarts": 2,
    "knn_candidates": 64,
    "rs_iters_retrieval": 20,
    "rs_iters_flat": 100,
    "rs_iters_rf": 200,
    "step_max": 0.30,
    "step_min": 0.02,
    "score_floor": 1e-3,
    "prefilter": 64,
    "sources": ["mlp_at", "gnn", "fagi_no_dp", "fagi", "knn"],
}
EPS = [str(e) for e in CONFIG["eps_sweep"]]
OLD_FAMILIES = ["mimicry32", "own_stored", "transfer_from_mlp", "transfer_from_gnn"]
SUITE_HASH = hashlib.sha1(json.dumps(SUITE, sort_keys=True).encode()).hexdigest()[:8]


def kind_of(name):
    if name in GRAPH_VARIANTS:
        return "graph"
    if name in FLAT_VARIANTS:
        return "flat"
    return name


class Retriever:
    """Exact k nearest pool flows: float32 candidate search, float64 re-ranking of the candidates."""

    def __init__(self, pool, k, k_inner):
        self.P = torch.as_tensor(pool)
        self.P64 = self.P.double()
        self.Pn = (self.P * self.P).sum(1)
        self.k, self.k_inner = k, k_inner

    def neighbours(self, Q, m=None, chunk=4000):
        m = m or self.k
        out_i, out_d = [], []
        for i in range(0, len(Q), chunk):
            q = Q[i:i + chunk]
            cand = (self.Pn[None] - 2.0 * (q @ self.P.T)).topk(max(SUITE["prefilter"], m), largest=False).indices
            diff = self.P64[cand] - q.double()[:, None, :]
            o = (diff * diff).sum(-1).topk(m, largest=False)
            out_i.append(cand.gather(1, o.indices))
            out_d.append(o.values)
        return torch.cat(out_i), torch.cat(out_d)

    def graph(self, Q, idx):
        nodes = torch.cat([Q[:, None, :], self.P[idx]], dim=1)
        n = nodes.shape[1]
        d = ((nodes[:, :, None, :] - nodes[:, None, :, :]) ** 2).sum(-1)
        eye = torch.eye(n, dtype=torch.bool)
        d = d.masked_fill(eye, float("inf"))
        A = torch.zeros(len(nodes), n, n)
        A[:, 0, 1:] = 1.0
        A[:, 1:, 0] = 1.0
        A.scatter_(2, d.topk(self.k_inner, largest=False).indices, 1.0)
        A = torch.maximum(A, A.transpose(1, 2))
        return nodes, A.masked_fill(eye, 0.0)


class Target:
    """One model on one fold: fast score, score with gradient, and the deployed-path score."""

    def __init__(self, folds, name, f, s):
        from robust_checks import scorer
        self.name, self.kind = name, kind_of(name)
        self.d = d = folds.fold(f)
        self.model = folds.load_model(name, f, s)
        self.true_score = scorer(folds, name, self.model, d)
        if self.kind in ("graph", "knn"):
            if "_retriever" not in d:
                d["_retriever"] = Retriever(d["Zr"], CONFIG["k"], CONFIG["k_inner"])
            self.R = d["_retriever"]
            self.yr = torch.as_tensor(d["yr"], dtype=torch.float32)

    @torch.no_grad()
    def score(self, Z):
        Zt = torch.as_tensor(np.ascontiguousarray(Z, dtype=np.float32))
        if self.kind == "rf":
            return self.model.predict_proba(inverse_flow(self.d["pp"], np.asarray(Z, dtype=np.float32)))[:, 1]
        if self.kind == "flat":
            lg = self.model(Zt[:, None, :], torch.zeros(len(Zt), 1, 1))
            return torch.sigmoid(lg[:, 1] - lg[:, 0]).numpy()
        out = []
        for i in range(0, len(Zt), 4000):
            q = Zt[i:i + 4000]
            idx, _ = self.R.neighbours(q)
            if self.kind == "knn":
                y = self.yr[idx]
                out.append(y.mean(1) + 1e-6 * (y[:, 0] - 0.5))
            else:
                lg = self.model(*self.R.graph(q, idx))
                out.append(torch.sigmoid(lg[:, 1] - lg[:, 0]))
        return torch.cat(out).numpy()

    def score_and_grad(self, Z):
        """Score to track and the gradient of the surrogate being minimized."""
        Zt = torch.as_tensor(np.ascontiguousarray(Z, dtype=np.float32))
        if self.kind == "flat":
            x = Zt.clone().requires_grad_(True)
            lg = self.model(x[:, None, :], torch.zeros(len(x), 1, 1))
            m = lg[:, 1] - lg[:, 0]
            (g,) = torch.autograd.grad(m.sum(), x)
            return torch.sigmoid(m).detach().numpy(), g.numpy()
        if self.kind == "graph":
            with torch.no_grad():
                idx, _ = self.R.neighbours(Zt)
                nodes, A = self.R.graph(Zt, idx)
            x = nodes.clone().requires_grad_(True)
            lg = self.model(x, A)
            m = lg[:, 1] - lg[:, 0]
            (g,) = torch.autograd.grad(m.sum(), x)
            return torch.sigmoid(m).detach().numpy(), g[:, 0, :].numpy()
        if self.kind == "knn":
            with torch.no_grad():
                idx, d2 = self.R.neighbours(Zt, SUITE["knn_candidates"])
                y = self.yr[idx]
                true = y[:, :self.R.k].mean(1) + 1e-6 * (y[:, 0] - 0.5)
                temp = (0.25 * d2[:, self.R.k - 1].float()).clamp(min=1e-3)
            x = Zt.clone().requires_grad_(True)
            dist = ((self.R.P[idx] - x[:, None, :]) ** 2).sum(-1)
            soft = (torch.softmax(-dist / temp[:, None], dim=1) * y).sum(1)
            (g,) = torch.autograd.grad(soft.sum(), x)
            return true.numpy(), g.numpy()
        raise ValueError(self.kind)


class Best:
    def __init__(self, Z0):
        self.z = Z0.copy()
        self.s = np.full(len(Z0), np.inf)

    def offer(self, Z, s, rows=None):
        rows = np.arange(len(self.s)) if rows is None else rows
        better = s < self.s[rows]
        self.z[rows[better]] = Z[better]
        self.s[rows[better]] = s[better]

    def active(self):
        return np.flatnonzero(self.s > SUITE["score_floor"])


def step_size(t, steps, eps):
    c = 0.5 * (1 + np.cos(np.pi * t / max(1, steps)))
    return eps * (SUITE["step_min"] + (SUITE["step_max"] - SUITE["step_min"]) * c)


def gradient_attack(tg, Z0, eps, best, steps, restarts, rng):
    for r in range(restarts):
        rows = best.active()
        if len(rows) == 0:
            return
        if r == 0:
            z = Z0[rows].copy()
        elif r == 1:
            z = best.z[rows].copy()
        else:
            z = (Z0[rows] + rng.uniform(-eps, eps, Z0[rows].shape)).astype(np.float32)
        for t in range(steps + 1):
            s, g = tg.score_and_grad(z)
            best.offer(z, s, rows)
            keep = s > SUITE["score_floor"]
            if not keep.any():
                break
            rows, z, g = rows[keep], z[keep], g[keep]
            if t == steps:
                break
            z = np.clip(z - step_size(t, steps, eps) * np.sign(g), Z0[rows] - eps, Z0[rows] + eps).astype(np.float32)


def random_search(tg, Z0, eps, best, iters, rng, mimic):
    d = Z0.shape[1]
    for t in range(iters):
        rows = best.active()
        if len(rows) == 0:
            return
        frac = 0.3 * (1 - t / iters) + 0.05
        mask = rng.random_sample((len(rows), d)) < frac
        cur = best.z[rows]
        if t % 2 == 0 or mimic is None:
            prop = Z0[rows] + eps * rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(len(rows), d))
        else:
            j = rng.randint(0, mimic.shape[1], size=len(rows))
            prop = Z0[rows] + np.clip(mimic[rows, j] - Z0[rows], -eps, eps)
        cand = np.where(mask, prop, cur).astype(np.float32)
        best.offer(cand, tg.score(cand), rows)


def _path(folds, stage, name, f, s, e):
    return os.path.join(CACHE_DIR, f"strong-{stage}-{folds.protocol}_{name}_f{f}_s{s}_e{e}_{folds.hash}_{SUITE_HASH}.npz")


def _load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _save(path, out):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def _mimic(d, Zm):
    if "_mimic" not in d:
        _, nb = d["benign_nn"].kneighbors(Zm)
        d["_mimic"] = d["benign_r"][nb]
    return d["_mimic"]


def base_scores(folds, name, f, s):
    """Deployed-path scores of benign test flows, benign pool flows (self excluded) and the clean attacked flows."""
    path = _path(folds, "base", name, f, s, "all")
    if os.path.exists(path):
        return _load(path)
    tg = Target(folds, name, f, s)
    d = tg.d
    pool_ben = np.flatnonzero(d["yr"] == 0)
    out = {"benign_test": tg.true_score(d["Zte"][d["yte"] == 0]),
           "benign_pool": tg.true_score(d["Zr"][pool_ben], self_index=pool_ben),
           "clean": tg.true_score(d["Zte"][d["mal"]])}
    _save(path, out)
    return out


def stage_one(folds, name, f, s, e):
    """Stored attacks re-scored, then the white-box score attack, at one budget. Cached."""
    path = _path(folds, "one", name, f, s, e)
    if os.path.exists(path):
        return _load(path)
    t0 = time.time()
    tg = Target(folds, name, f, s)
    d = tg.d
    Zm = d["Zte"][d["mal"]]
    mimic = _mimic(d, Zm)
    blob = folds.result(name, f, s)
    rng = np.random.RandomState(15000 + 100 * f + s + int(round(1000 * float(e))))
    eps = float(e)
    best = Best(Zm)
    out = {}
    cand = (Zm[:, None, :] + np.clip(mimic - Zm[:, None, :], -eps, eps)).astype(np.float32)
    sm = tg.score(cand.reshape(-1, Zm.shape[1])).reshape(len(Zm), -1)
    out["mimicry32"] = sm.min(1)
    best.offer(cand[np.arange(len(Zm)), sm.argmin(1)], out["mimicry32"])
    del cand
    stored = {}
    if e in blob["adv_examples"]:
        stored["own_stored"] = blob["adv_examples"][e]
    for src in TRANSFER_SOURCES:
        if src != name:
            stored[f"transfer_from_{src}"] = folds.result(src, f, s)["adv_examples"][e]
    if tg.kind == "graph":
        xa = adversarial_inputs(tg.model, d["Gte"][d["mal"]], d["Ate"][d["mal"]], np.ones(len(Zm), dtype=np.int64),
                                node_mask=center_mask(len(Zm), d["Gte"].shape[1]), eps=eps,
                                steps=CONFIG["pgd_eval_steps"], seed=7000 + 100 * f + s)
        stored["pgd_reretrieved"] = xa[:, 0, :]
    for k_, ex in stored.items():
        ex = np.asarray(ex, dtype=np.float32)
        out[k_] = tg.score(ex)
        best.offer(ex, out[k_])
    if tg.kind in ("graph", "flat", "knn"):
        steps, restarts = {"graph": (SUITE["graph_steps"], SUITE["graph_restarts"]),
                           "flat": (SUITE["flat_steps"], SUITE["flat_restarts"]),
                           "knn": (SUITE["knn_steps"], SUITE["knn_restarts"])}[tg.kind]
        own = Best(Zm)
        own.offer(best.z, best.s)
        gradient_attack(tg, Zm, eps, own, steps, restarts, rng)
        out["score_attack"] = own.s.copy()
        best = own
    out = {k_: np.asarray(v, dtype=np.float32) for k_, v in out.items()}
    out["best_z"] = best.z
    out["best_s"] = best.s.astype(np.float32)
    out["seconds"] = np.array(time.time() - t0)
    _save(path, out)
    print(f"    [{folds.protocol}] stage1 {name:<14} fold {f} seed {s} eps {e}: default-threshold recall "
          f"{np.mean(out['best_s'] > 0.5):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def sources_for(name):
    return [m for m in SUITE["sources"] if m != name]


def suite(folds, name, f, s, e):
    """Full suite for one model at one budget: per-flow lowest deployed-path score. Cached."""
    path = _path(folds, "full", name, f, s, e)
    if os.path.exists(path):
        return _load(path)
    one = stage_one(folds, name, f, s, e)
    t0 = time.time()
    tg = Target(folds, name, f, s)
    d = tg.d
    Zm = d["Zte"][d["mal"]]
    mimic = _mimic(d, Zm)
    rng = np.random.RandomState(16000 + 100 * f + s + int(round(1000 * float(e))))
    iters = {"rf": SUITE["rs_iters_rf"], "flat": SUITE["rs_iters_flat"]}.get(tg.kind, SUITE["rs_iters_retrieval"])
    eps = float(e)
    out = {k: v for k, v in one.items() if k not in ("best_z", "best_s", "seconds")}
    best = Best(Zm)
    best.offer(one["best_z"], one["best_s"].astype(np.float64))
    tr = np.full(len(Zm), np.inf)
    for src in sources_for(name):
        ex = stage_one(folds, src, f, s, e)["best_z"]
        sc = tg.score(ex)
        tr = np.minimum(tr, sc)
        best.offer(ex, sc)
    out["transfer_strong"] = tr.astype(np.float32)
    before = best.s.copy()
    random_search(tg, Zm, eps, best, iters, rng, mimic)
    out["random_search"] = best.s.astype(np.float32)
    out["random_search_gain"] = (before - best.s).astype(np.float32)
    assert np.abs(best.z - Zm).max() <= eps + 1e-5
    out["final"] = tg.true_score(best.z).astype(np.float64)
    out["best_z"] = best.z
    out["seconds"] = np.array(float(one["seconds"]) + time.time() - t0)
    _save(path, out)
    print(f"    [{folds.protocol}] suite  {name:<14} fold {f} seed {s} eps {e}: default-threshold recall "
          f"{np.mean(out['final'] > 0.5):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def have(folds, name, f, s, e):
    return os.path.exists(_path(folds, "full", name, f, s, e))


def rescored(sc):
    """Lowest score over the stored attacks that the earlier matched-FPR analysis re-scored."""
    return np.min(np.stack([sc[k] for k in OLD_FAMILIES if k in sc]), axis=0)


def worst_detected(folds, name, f, s, eps_list=None):
    """Per-flow detection at the default threshold: the stored attacks and the full suite must all fail."""
    old = folds.worst_detected(name, f, s)
    return {e: old[e] & (suite(folds, name, f, s, e)["final"] > 0.5) for e in (eps_list or EPS)}


MAIN = ["rf", "knn", "mlp", "mlp_at", "mlp_dp", "cnn", "gnn", "fagi_no_dp", "fagi"]
EXTRA = ["fagi_no_guard", "fagi_no_pgd"] + [f"fagi_sigma{x}" for x in CONFIG["sigma_sweep"]]
TEMPORAL = ["rf", "knn", "mlp", "mlp_at", "mlp_dp", "gnn", "fagi_no_dp", "fagi"]


SHORT_EPS = ["0.1", "0.5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--protocol", choices=["cv", "temporal"], default="cv")
    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["seeds"])
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--extra", action="store_true", help="also the ablation and sigma-sweep variants (seed 0)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    folds = CICIDSFolds(args.data_path, protocol=args.protocol)
    print(f"protocol {args.protocol}, config {folds.hash}, suite {SUITE_HASH}", flush=True)
    for f in (args.folds if args.folds is not None else range(folds.n_splits)):
        for s in args.seeds:
            main_names = args.models or (MAIN if args.protocol == "cv" else TEMPORAL)
            eps_main = EPS if args.protocol == "cv" else SHORT_EPS
            jobs = [(m, e) for e in eps_main for m in main_names]
            if args.extra and s == 0 and args.protocol == "cv" and not args.models:
                jobs += [(m, e) for e in SHORT_EPS for m in EXTRA]
            for m, e in jobs:
                if have(folds, m, f, s, e):
                    continue
                base_scores(folds, m, f, s)
                for src in SUITE["sources"]:
                    stage_one(folds, src, f, s, e)
                suite(folds, m, f, s, e)


if __name__ == "__main__":
    main()
