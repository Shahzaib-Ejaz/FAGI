"""Checks on the strength of the attack suite and on what the matched-FPR numbers measure (CICIDS2017, eps 0.5).

Run after strong_attacks.py; reads its cached best examples and adds, from those starting points:
  1. Longer score-based random search (1,000 queries per flow) on the pool-based models (seed 0, all folds).
  2. A joint attack on the graph models: the logit-margin gradient plus three times the neighborhood-attack
     gradient (both L1-normalized), neighbors re-retrieved at every step, 20 steps, from the best example and
     from one random start (seed 0, all folds).
  Both stop on a flow once it falls below the model's matched 1% threshold and report matched recall before
  and after.
  3. A certified lower bound for the k-NN vote: a flow is certainly still detected if, anywhere in its L-inf box,
     at least 6 of its 8 nearest pool flows are malicious. Sufficient condition: the 6th smallest upper bound on
     the distance to a malicious pool flow is below the smallest lower bound on the distance to a benign one.
  4. Recall at the matched rate per attack label (all folds and seeds), and recall restricted to the labels that
     occur in the test portion of the file-order split.
  5. How much the k-NN vote's matched recall depends on random tie-breaking (200 draws per fold).
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import strong_attacks as sa
from cicids_folds import CONFIG, CICIDSFolds
from cicids_stats import ms
from common import env_info, save_json

E = "0.5"
POOL_MODELS = ["knn", "gnn", "fagi_no_dp", "fagi"]


def threshold(bt, discrete, rng):
    js = 1e-7 if discrete else 1e-15
    return float(np.quantile(bt + rng.uniform(0, js, len(bt)), 0.99)), js


def recall(sc, thr, js, seed=1):
    return float(np.mean(sc + np.random.RandomState(seed).uniform(0, js, len(sc)) > thr))


def stronger(folds, name, f, mode):
    tg = sa.Target(folds, name, f, 0)
    d = tg.d
    Zm = d["Zte"][d["mal"]]
    eps = float(E)
    full = sa.suite(folds, name, f, 0, E)
    base = sa.base_scores(folds, name, f, 0)
    thr, js = threshold(base["benign_test"], name == "knn", np.random.RandomState(0))
    best = sa.Best(Zm, thr - 1e-9)
    best.offer(full["best_z"].astype(np.float32), full["final"])
    before = recall(full["final"], thr, js)
    if mode == "random_search":
        sa.random_search(tg, Zm, eps, best, 1000, np.random.RandomState(5), sa._mimic(d, Zm), ())
    else:
        R = tg.R
        for r in range(2):
            rows = np.flatnonzero(best.s > thr)
            if len(rows) == 0:
                break
            Z0r = Zm[rows]
            z = best.z[rows].copy() if r == 0 else (Z0r + np.random.RandomState(200 + r).uniform(-eps, eps, Z0r.shape)).astype(np.float32)
            for t in range(21):
                zt = torch.as_tensor(z)
                with torch.no_grad():
                    idx, _ = R.neighbours(zt)
                    nodes, A = R.graph(zt, idx)
                    ib, im, _ = R.by_class(zt, SUITE_NB, SUITE_NM)
                x = nodes.clone().requires_grad_(True)
                lg = tg.model(x, A)
                m = lg[:, 1] - lg[:, 0]
                (g1,) = torch.autograd.grad(m.sum(), x)
                xc = zt.clone().requires_grad_(True)
                db = ((R.P[ib] - xc[:, None, :]) ** 2).sum(-1).clamp(min=1e-12).sqrt().sum(1)
                dm = ((R.P[im] - xc[:, None, :]) ** 2).sum(-1).clamp(min=1e-12).sqrt().sum(1)
                (g2,) = torch.autograd.grad((db - dm).sum(), xc)
                best.offer(z.copy(), torch.sigmoid(m.detach().double()).numpy(), rows)
                if t == 20:
                    break
                g1 = g1[:, 0, :]
                g = g1 / (g1.abs().sum(1, keepdim=True) + 1e-12) + 3.0 * g2 / (g2.abs().sum(1, keepdim=True) + 1e-12)
                z = np.clip(z - sa.step_size(t, 20, eps) * np.sign(g.numpy()), Z0r - eps, Z0r + eps).astype(np.float32)
    assert np.abs(best.z - Zm).max() <= eps + 1e-5
    after = recall(np.asarray(tg.true_score(best.z), dtype=np.float64), thr, js)
    print(f"  {mode:<14} {name:<11} fold {f}: matched recall {before:.4f} -> {after:.4f}", flush=True)
    return before, after


SUITE_NB, SUITE_NM = sa.SUITE["nbr_benign"], sa.SUITE["nbr_malicious"]


def knn_certificate(folds, f, eps):
    d = folds.fold(f)
    Zm = torch.as_tensor(d["Zte"][d["mal"]]).float()
    P = torch.as_tensor(d["Zr"]).float()
    ben, mal = P[torch.as_tensor(d["yr"] == 0)], P[torch.as_tensor(d["yr"] == 1)]
    ub, lb = [], []
    for i in range(0, len(Zm), 40):
        x = Zm[i:i + 40]
        ub.append(((x[:, None, :] - mal[None]).abs() + eps).pow(2).sum(-1).topk(6, largest=False).values[:, 5])
        lb.append(((x[:, None, :] - ben[None]).abs() - eps).clamp(min=0).pow(2).sum(-1).min(1).values)
    return float((torch.cat(ub) < torch.cat(lb)).double().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--output", default="experiments/results/attack_checks.json")
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    folds = CICIDSFolds(args.data_path)
    out = {"env": env_info(), "eps": E, "random_search_queries": 1000, "joint": {"steps": 20, "starts": 2, "weight": 3.0}}

    # 3. certificate and 5. tie-breaking for the vote
    cert = {e: [knn_certificate(folds, f, float(e)) for f in range(folds.n_splits)] for e in ("0.1", "0.5")}
    print("k-NN certified:", {e: np.round(v, 4).tolist() for e, v in cert.items()}, flush=True)
    out["knn_certified"] = {e: ms(v) for e, v in cert.items()}
    sds, ranges = [], []
    for f in range(folds.n_splits):
        base, full = sa.base_scores(folds, "knn", f, 0), sa.suite(folds, "knn", f, 0, E)
        vals = []
        for r in range(200):
            rng = np.random.RandomState(50000 + r)
            thr = np.quantile(base["benign_test"] + rng.uniform(0, 1e-7, len(base["benign_test"])), 0.99)
            vals.append(np.mean(full["final"] + rng.uniform(0, 1e-7, len(full["final"])) > thr))
        sds.append(float(np.std(vals)))
        ranges.append(float(np.max(vals) - np.min(vals)))
    out["knn_tiebreak"] = {"sd": ms(sds), "range": ms(ranges)}

    # 4. per-label recall at the matched rate
    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp5_temporal.json")) as fh:
        temporal_test_labels = {k for k, v in json.load(fh)["label_counts"].items() if v.get("test", 0) > 0}
    per = {m: collections.Counter() for m in POOL_MODELS}
    n = collections.Counter()
    restricted = {m: [] for m in POOL_MODELS}
    for f in range(folds.n_splits):
        d = folds.fold(f)
        lab = d["labels_te"][d["mal"]]
        keep = np.isin(lab, list(temporal_test_labels))
        for s in CONFIG["seeds"]:
            for l in lab:
                n[l] += 1
            for m in POOL_MODELS:
                base, full = sa.base_scores(folds, m, f, s), sa.suite(folds, m, f, s, E)
                thr, js = threshold(base["benign_test"], m == "knn", np.random.RandomState(12000 + 100 * f + s))
                det = full["final"] + np.random.RandomState(1).uniform(0, js, len(full["final"])) > thr
                for l in lab[det]:
                    per[m][l] += 1
                restricted[m].append(float(det[keep].mean()))
    out["per_label"] = {l: {"n": int(n[l]), **{m: per[m][l] / n[l] for m in POOL_MODELS}} for l in sorted(n, key=lambda x: -n[x])}
    dos = ["DoS Hulk", "DoS Slowhttptest", "DoS slowloris"]
    out["share_detected_from_dos3"] = {m: sum(per[m][l] for l in dos) / max(1, sum(per[m].values())) for m in POOL_MODELS}
    out["restricted_to_file_order_test_labels"] = {"labels": sorted(temporal_test_labels), **{m: ms(v) for m, v in restricted.items()}}
    out["dos3_in_file_order_test"] = sorted(set(dos) & temporal_test_labels)
    save_json(out, args.output)

    # 1. and 2. stronger attacks, seed 0
    for mode, models in (("random_search", POOL_MODELS), ("joint", ["gnn", "fagi_no_dp", "fagi"])):
        res = {m: [stronger(folds, m, f, mode) for f in range(folds.n_splits)] for m in models}
        out[mode] = {m: {"before": ms([b for b, _ in v]), "after": ms([a for _, a in v]),
                         "max_drop": max(b - a for b, a in v)} for m, v in res.items()}
        save_json(out, args.output)


if __name__ == "__main__":
    main()
