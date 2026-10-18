import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cicids_folds import CONFIG, CICIDSFolds
from cicids_stats import ms, paired, wilson
from common import env_info, mcnemar, metrics, save_json

MODELS = [
    ("Random Forest", "rf"),
    ("k-NN vote (k=8)", "knn"),
    ("MLP", "mlp"),
    ("MLP + PGD training", "mlp_at"),
    ("MLP + PGD + DP-SGD, all data private", "mlp_dp"),
    ("1D-CNN", "cnn"),
    ("GNN Baseline", "gnn"),
    ("FAGI", "fagi"),
]
DETERMINISTIC = {"knn"}


def fold_means(vals, n_folds, n_seeds):
    v = np.asarray(vals, float).reshape(n_folds, n_seeds)
    return v.mean(axis=1).tolist()


def compare(per, a, b, field, n_folds, n_seeds):
    if a in DETERMINISTIC or b in DETERMINISTIC:
        out = paired(fold_means(per[a][field], n_folds, n_seeds), fold_means(per[b][field], n_folds, n_seeds))
        out["unit"] = "fold (seeds averaged)"
        return out
    out = paired(per[a][field], per[b][field])
    out["unit"] = "fold-seed pair"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--output", default="experiments/results/exp1.json")
    args = ap.parse_args()
    print("EXPERIMENT 1: CICIDS2017 benchmark (5-fold CV; private training set + public reference pool)")
    folds = CICIDSFolds(args.data_path)
    seeds = CONFIG["seeds"]
    nf = folds.n_splits
    per = {k: {"acc": [], "fpr": [], "dr": [], "f1": [], "eps": [], "w01": [], "w05": []} for _, k in MODELS}
    pooled_pred = {k: [] for _, k in MODELS}
    pooled_lab, pooled_y = [], []
    for f in range(nf):
        d = folds.fold(f)
        for s in seeds:
            for _, key in MODELS:
                b = folds.result(key, f, s)
                w = folds.worst_detected(key, f, s)
                m = metrics(d["yte"], b["pred_clean"])
                p = per[key]
                p["acc"].append(m["accuracy"]); p["fpr"].append(m["fpr"]); p["dr"].append(m["detection_rate"])
                p["f1"].append(m["f1"]); p["eps"].append(b["info"].get("epsilon"))
                p["w01"].append(float(w["0.1"].mean())); p["w05"].append(float(w["0.5"].mean()))
                if s == seeds[0]:
                    pooled_pred[key].append(b["pred_clean"])
            if s == seeds[0]:
                pooled_lab.append(d["labels_te"]); pooled_y.append(d["yte"])
    lab, y = np.concatenate(pooled_lab), np.concatenate(pooled_y)
    rows = []
    for name, key in MODELS:
        p = per[key]
        pred = np.concatenate(pooled_pred[key])
        pc = {}
        for l in sorted(set(lab)):
            msk = lab == l
            want = 0 if l == "BENIGN" else 1
            k_, n_ = int((pred[msk] == want).sum()), int(msk.sum())
            pc[l] = {"recall": k_ / n_, "k": k_, "n": n_, "ci95": wilson(k_, n_)}
        stat = (lambda v: ms(fold_means(v, nf, len(seeds)))) if key in DETERMINISTIC else ms
        row = {"name": name, "key": key, "deterministic_across_seeds": key in DETERMINISTIC,
               "clean": stat(p["acc"]), "fpr": stat(p["fpr"]), "detection_rate": stat(p["dr"]), "f1": stat(p["f1"]),
               "worst_recall_eps0.1": stat(p["w01"]), "worst_recall_eps0.5": stat(p["w05"]),
               "dp": key in ("fagi", "mlp_dp"),
               "epsilon": ms(p["eps"]) if any(e is not None for e in p["eps"]) else None, "per_class_seed0": pc}
        rows.append(row)
        print(f"  {name:<40} clean {row['clean']['mean'] * 100:6.2f}  FPR {row['fpr']['mean'] * 100:5.2f}  "
              f"worst@0.1 {row['worst_recall_eps0.1']['mean'] * 100:6.2f}  @0.5 {row['worst_recall_eps0.5']['mean'] * 100:6.2f}")
    fp = {}
    for f in range(nf):
        b = folds.result("gnn_fullpool", f, 0)
        fp.setdefault("acc", []).append(float(np.mean(b["pred_clean"] == folds.fold(f)["yte"])))
    tests = {f"{a}_vs_{b}_{fld}": compare(per, a, b, fld, nf, len(seeds))
             for a, b, fld in [("fagi", "gnn", "acc"), ("fagi", "gnn", "w01"), ("fagi", "gnn", "w05"),
                               ("fagi", "knn", "w05"), ("fagi", "mlp_dp", "w05"), ("fagi", "mlp_dp", "acc"),
                               ("rf", "fagi", "acc"), ("knn", "gnn", "w05")]}
    tests["mcnemar_fagi_vs_gnn_clean_seed0"] = mcnemar(y, np.concatenate(pooled_pred["gnn"]),
                                                       np.concatenate(pooled_pred["fagi"]))
    d0 = folds.fold(0)
    out = {"config": CONFIG, "config_hash": folds.hash, "env": env_info(),
           "dataset": {k: v for k, v in folds.meta.items() if k != "features"},
           "n_features": len(folds.features), "features": folds.features,
           "split": {"private_train_per_fold": int(len(d0["priv"])), "reference_pool_per_fold": int(len(d0["pool"])),
                     "test_per_fold": int(len(d0["te"])), "attacked_malicious_per_fold": int(len(d0["mal"]))},
           "gnn_fullpool_seed0": {"clean": ms(fp["acc"]),
                                  "note": "non-private GNN trained on the whole training fold, neighbours from the whole fold"},
           "models": rows, "tests": tests}
    save_json(out, args.output)


if __name__ == "__main__":
    main()
