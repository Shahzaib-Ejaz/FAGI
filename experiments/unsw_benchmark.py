"""The main CICIDS2017 comparison repeated on UNSW-NB15 (5 folds x CONFIG seeds by default; --seeds 0 for one seed; attack budgets 0.1 and 0.5).

Same protocol as CICIDS2017: stratified sample (at most 3,000 flows per attack category, as many benign flows as
attack flows), 5-fold cross-validation, each training fold split into a private half and a public reference pool,
the same models, the same score-minimizing attack suite (strong_attacks.py) and the same evaluation
(robust_checks.evaluate: worst-case recall at a matched false-positive rate on benign test flows).
Caches use the protocol name "unsw-cv", so nothing here touches the CICIDS2017 results.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import StratifiedKFold

import robust_checks as rc
import strong_attacks as sa
from cicids_folds import CONFIG, CICIDSFolds, config_hash
from cicids_stats import ms, paired_folds
from common import env_info, metrics, save_json
from data.unsw_nb15 import load_unsw_nb15

EPS_LIST = ["0.1", "0.5"]
MODELS = ["rf", "knn", "mlp", "mlp_at", "mlp_dp", "cnn", "gnn", "fagi_no_dp", "fagi"]


class UNSWFolds(CICIDSFolds):
    def __init__(self, data_path: str):
        X, y, labels, feats, meta = load_unsw_nb15(data_path, CONFIG["per_class_cap"], CONFIG["sample_seed"])
        self.X, self.y, self.labels, self.features, self.meta = X, y, labels, feats, meta
        self.protocol = "unsw-cv"
        skf = StratifiedKFold(CONFIG["n_folds"], shuffle=True, random_state=CONFIG["cv_seed"])
        self.splits = list(skf.split(X, labels))
        self.n_splits = len(self.splits)
        self._prepared = {}
        self.hash = config_hash()


def run(folds, models, seeds, fold_list):
    rows = {m: [] for m in models}
    for s in seeds:  # seed-major, so seed 0 on every fold finishes first
        for f in fold_list:
            for m in models:
                d = folds.fold(f)
                b = folds.result(m, f, s)
                sc = rc.scores_for(folds, m, f, s, EPS_LIST)
                row = rc.evaluate(sc, np.random.RandomState(12000 + 100 * f + s), EPS_LIST)
                mt = metrics(d["yte"], b["pred_clean"])
                row.update({"fold": f, "seed": s, "clean": mt["accuracy"], "fpr": mt["fpr"],
                            "epsilon": (b["info"] or {}).get("epsilon")})
                rows[m].append(row)
                print(f"  [unsw] {m:<11} fold {f} seed {s}: clean {row['clean']:.4f} FPR {row['fpr']:.4f}  matched 1%: "
                      f"0.1 {row['fixed_0.01_0.1']:.3f}  0.5 {row['fixed_0.01_0.5']:.3f}  default 0.5 {row['default_0.5']:.3f}",
                      flush=True)
    for m in rows:
        rows[m].sort(key=lambda x: (x["fold"], x["seed"]))  # fold-major, as summarize and paired_folds expect
    return rows


def summarize(rows, n_folds):
    keys = ["clean", "fpr", "fpr_default", "fixed_0.01_0.1", "fixed_0.01_0.5", "fixed_0.02_0.5", "default_0.1", "default_0.5", "epsilon"]
    out = {}
    for m, r in rows.items():
        out[m] = {}
        for k in keys:
            v = [x.get(k) for x in r]
            if all(isinstance(x, (int, float)) and x is not None for x in v):
                fold_means = np.asarray(v, float).reshape(n_folds, -1).mean(axis=1)
                out[m][k] = ms(fold_means.tolist())
        out[m]["per_run"] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/unsw_nb15/")
    ap.add_argument("--output", default="experiments/results/unsw_benchmark.json")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(CONFIG["seeds"]))
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    folds = UNSWFolds(args.data_path)
    fold_list = args.folds if args.folds is not None else list(range(folds.n_splits))
    print(f"UNSW-NB15: {folds.meta['sample_rows']} flows, {len(folds.features)} features, config {folds.hash}", flush=True)
    rows = run(folds, args.models, args.seeds, fold_list)
    nf = len(fold_list)
    res = summarize(rows, nf)
    T = {}
    pairs = [("fagi_no_dp", "gnn"), ("fagi_no_dp", "fagi"), ("fagi", "gnn"), ("knn", "fagi"), ("knn", "fagi_no_dp"), ("mlp_at", "fagi")]
    for a, b in pairs:
        if a in rows and b in rows:
            for k in ("fixed_0.01_0.1", "fixed_0.01_0.5", "clean"):
                T[f"{a}_vs_{b}_{k}"] = paired_folds([x[k] for x in rows[a]], [x[k] for x in rows[b]], nf)
    meta = {k: v for k, v in folds.meta.items() if k != "features"}
    save_json({"env": env_info(), "config_hash": folds.hash, "seeds": args.seeds, "folds": fold_list, "eps": EPS_LIST,
               "dataset": meta, "features": folds.features, "models": res, "tests": T}, args.output)


if __name__ == "__main__":
    main()
