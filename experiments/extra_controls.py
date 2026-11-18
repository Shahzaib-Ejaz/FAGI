"""Controls added after review, on CICIDS2017 (seed 0, 5 folds, attack budgets 0.1 and 0.5).

1. What DP-SGD costs, separated from what clipping and the training configuration cost (standard split):
     fagi_clip_only   FAGI trained with DP-SGD's per-example clipping and Poisson sampling but no noise (sigma = 0,
                      so no privacy guarantee).
     fagi_dp_tuned    FAGI with a configuration chosen for private training instead of the non-private one:
                      expected batch 1,024, 40 epochs, learning rate 0.008, and the noise multiplier that gives the
                      same epsilon (2.90 at delta = 1e-6) at the new sampling rate and step count.
2. Models with epsilon = 0 with respect to the private set: trained on the public reference pool alone, with each
   pool flow excluded from its own neighborhood (protocol cv-poolonly). The k-NN vote is already such a model.
3. A smaller pool: 10% of each training fold is the pool and 90% is private (protocol cv-pool10), together with
   the same models trained on that 10% pool alone (cv-pool10-poolonly).

Every model is attacked with the full score-minimizing suite of strong_attacks.py and evaluated as in
robust_checks.py (worst-case recall at a matched 1% false-positive rate on benign test flows).
Caches use the protocol name, so nothing here touches the main results.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import cicids_folds as cf
import robust_checks as rc
from cicids_folds import CONFIG, GRAPH_VARIANTS, CICIDSFolds
from cicids_stats import ms, paired_folds
from common import env_info, metrics, save_json
from data.cicids2017 import EgoGraphBuilder, FlowPreprocessor, center_mask
from fagi.privacy.dp_sgd import compute_epsilon
from fagi.training import train_model

TUNED = {"batch_size": 1024, "epochs": 40, "lr": 0.008}
EPS_LIST = ["0.1", "0.5"]


def tuned_sigma(n, target):
    q = TUNED["batch_size"] / n
    steps = math.ceil(TUNED["epochs"] / q)
    lo, hi = 0.3, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if compute_epsilon(mid, q, steps, CONFIG["dp_delta"]) > target else (lo, mid)
    return hi


GRAPH_VARIANTS["fagi_clip_only"] = {"guard": True, "adv": True, "dp": True, "seed": 16, "sigma": 0.0}
GRAPH_VARIANTS["fagi_dp_tuned"] = {"guard": True, "adv": True, "dp": True, "seed": 16, "tuned": True}


class Folds(CICIDSFolds):
    """CICIDSFolds with a different pool fraction and/or training on the pool instead of the private set."""

    def __init__(self, data_path, pool_fraction=None, train_on="private"):
        super().__init__(data_path, "cv")
        self.pool_fraction, self.train_on = pool_fraction, train_on
        tag = []
        if pool_fraction is not None:
            tag.append(f"pool{int(round(100 * pool_fraction))}")
        if train_on == "pool":
            tag.append("poolonly")
        if tag:
            self.protocol = "cv-" + "-".join(tag)

    def fold(self, f):
        if self.pool_fraction is None and self.train_on == "private":
            return super().fold(f)
        if f in self._prepared:
            return self._prepared[f]
        tr, te = self.splits[f]
        frac = self.pool_fraction if self.pool_fraction is not None else CONFIG["pool_fraction"]
        try:
            priv, pool = train_test_split(tr, test_size=frac, random_state=1000 + f, stratify=self.labels[tr])
        except ValueError:
            priv, pool = train_test_split(tr, test_size=frac, random_state=1000 + f)
        priv, pool = np.sort(priv), np.sort(pool)
        pp = FlowPreprocessor().fit(self.X[pool])
        Zr, Zte = pp.transform(self.X[pool]), pp.transform(self.X[te])
        eb = EgoGraphBuilder(CONFIG["k"], CONFIG["k_inner"]).fit(Zr)
        yr = self.y[pool]
        if self.train_on == "pool":
            # Train on the pool itself; each pool flow is excluded from its own neighborhood, as at calibration.
            train_idx, Zp = pool, Zr
            Gp, Ap, _ = eb.build(Zr, self_index=np.arange(len(Zr)))
        else:
            train_idx, Zp = priv, pp.transform(self.X[priv])
            Gp, Ap, _ = eb.build(Zp)
        Gte, Ate, nte = eb.build(Zte)
        mal_all = np.flatnonzero(self.y[te] == 1)
        mal = np.sort(np.random.RandomState(2000 + f).choice(mal_all, min(CONFIG["attack_n"], len(mal_all)), replace=False))
        benign_nn = NearestNeighbors(n_neighbors=CONFIG["mimicry_targets"], algorithm="brute", n_jobs=-1).fit(Zr[yr == 0])
        d = {"tr": tr, "priv": train_idx, "pool": pool, "te": te, "pp": pp, "Zp": Zp, "Zr": Zr, "Zte": Zte, "eb": eb,
             "Gp": Gp, "Ap": Ap, "Gte": Gte, "Ate": Ate, "nbr_te": nte, "yr": yr,
             "Mp": center_mask(len(Gp), Gp.shape[1]), "yp": self.y[train_idx], "yte": self.y[te],
             "labels_te": self.labels[te], "labels_p": self.labels[train_idx], "mal": mal,
             "benign_r": Zr[yr == 0], "benign_nn": benign_nn, "n_private": len(priv), "n_pool": len(pool)}
        self._prepared = {f: d}
        return d

    def _train(self, name, f, s, d):
        v = GRAPH_VARIANTS.get(name, {})
        if not v.get("tuned"):
            return super()._train(name, f, s, d)
        model = self._build_model(name, d)
        n = len(d["yp"])
        info = train_model(model, d["Gp"], d["Ap"], d["yp"], node_mask=d["Mp"], epochs=TUNED["epochs"],
                           batch_size=TUNED["batch_size"], lr=TUNED["lr"], adv=True, dp=True,
                           sigma=tuned_sigma(n, compute_epsilon(CONFIG["dp_sigma"], CONFIG["batch_size"] / n,
                                                                math.ceil(CONFIG["epochs"] / (CONFIG["batch_size"] / n)),
                                                                CONFIG["dp_delta"])),
                           clip=CONFIG["dp_clip"], delta=CONFIG["dp_delta"], seed=1000 * f + 100 * s + v["seed"],
                           pgd_eps=CONFIG["pgd_eps"], pgd_steps=CONFIG["pgd_train_steps"])
        return model, info


def evaluate(folds, names, fold_list=None):
    out = {}
    for m in names:
        rows = []
        for f in (fold_list if fold_list is not None else range(folds.n_splits)):
            d = folds.fold(f)
            b = folds.result(m, f, 0)
            sc = rc.scores_for(folds, m, f, 0, EPS_LIST)
            row = rc.evaluate(sc, np.random.RandomState(12000 + 100 * f), EPS_LIST)
            mt = metrics(d["yte"], b["pred_clean"])
            row["fold"] = f
            eps_dp = (b["info"] or {}).get("epsilon")
            eps_dp = eps_dp if eps_dp is not None and math.isfinite(eps_dp) else None  # sigma = 0: no guarantee
            row.update({"clean": mt["accuracy"], "fpr": mt["fpr"], "epsilon": eps_dp,
                        "sigma": (b["info"] or {}).get("sigma"), "n_train": int(len(d["yp"])), "n_pool": int(len(d["yr"]))})
            rows.append(row)
            print(f"  [{folds.protocol}] {m:<15} fold {f}: clean {row['clean']:.4f} FPR {row['fpr']:.4f}  matched 1%: "
                  f"0.1 {row['fixed_0.01_0.1']:.3f}  0.5 {row['fixed_0.01_0.5']:.3f}", flush=True)
        out[m] = {k: ms([r[k] for r in rows]) for k in rows[0] if isinstance(rows[0][k], (int, float)) and rows[0][k] is not None}
        out[m]["per_fold"] = rows
    return out


def tests(res, pairs, key):
    return {f"{a}_vs_{b}_{key}": paired_folds([r[key] for r in res[a]["per_fold"]], [r[key] for r in res[b]["per_fold"]], len(res[a]["per_fold"]))
            for a, b in pairs if a in res and b in res}


def pool_overlap(data_path):
    """How close private and test flows are to the published pool (L-infinity, standardized log features)."""
    F = CICIDSFolds(data_path)
    rows = []
    for f in range(F.n_splits):
        d = F.fold(f)
        nn = NearestNeighbors(n_neighbors=1, metric="chebyshev", algorithm="brute", n_jobs=-1).fit(d["Zr"])
        r = {}
        for tag, Z in (("private", d["Zp"]), ("test", d["Zte"])):
            dist = nn.kneighbors(Z)[0][:, 0]
            for t in (0.1, 0.5):
                r[f"{tag}_within_{t}"] = float(np.mean(dist <= t))
            r[f"{tag}_median"] = float(np.median(dist))
        rows.append(r)
        print(f"  overlap fold {f}: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
    return {k: ms([r[k] for r in rows]) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--output", default="experiments/results/extra_controls.json")
    ap.add_argument("--parts", nargs="*", default=["dp", "poolonly", "pool10", "overlap"])
    ap.add_argument("--pool_folds", type=int, nargs="*", default=[0, 1, 2], help="folds for the pool-only and 10%%-pool protocols")
    args = ap.parse_args()
    out = {"env": env_info(), "tuned": TUNED, "eps": EPS_LIST, "seed": 0}
    if os.path.exists(args.output):
        import json
        with open(args.output) as fh:
            out.update({k: v for k, v in json.load(fh).items() if k not in ("env",)})
    main_names = ["knn", "gnn", "fagi_no_dp", "fagi", "rf"]
    if "dp" in args.parts:
        F = Folds(args.data_path)
        r = evaluate(F, ["fagi_clip_only", "fagi_dp_tuned", "fagi", "fagi_no_dp"])
        out["dp_controls"] = {"models": r, "tests": {**tests(r, [("fagi_clip_only", "fagi"), ("fagi_dp_tuned", "fagi"), ("fagi_no_dp", "fagi_clip_only")], "fixed_0.01_0.5"),
                                                    **tests(r, [("fagi_clip_only", "fagi"), ("fagi_dp_tuned", "fagi"), ("fagi_no_dp", "fagi_clip_only")], "clean")}}
        save_json(out, args.output)
    if "poolonly" in args.parts:
        r = evaluate(Folds(args.data_path, train_on="pool"), ["rf", "gnn", "fagi_no_dp"], args.pool_folds)
        out["pool_only"] = {"models": r}
        save_json(out, args.output)
    if "overlap" in args.parts:
        out["pool_overlap"] = pool_overlap(args.data_path)
        save_json(out, args.output)
    if "pool10" in args.parts:
        r = evaluate(Folds(args.data_path, pool_fraction=0.1), main_names, args.pool_folds)
        r0 = evaluate(Folds(args.data_path, pool_fraction=0.1, train_on="pool"), ["rf", "gnn", "fagi_no_dp"], args.pool_folds)
        out["pool10"] = {"models": r, "pool_only": r0}
        save_json(out, args.output)


if __name__ == "__main__":
    main()
