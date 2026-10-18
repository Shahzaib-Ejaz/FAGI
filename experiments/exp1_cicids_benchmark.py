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
    ("1D-CNN", "cnn"),
    ("GNN Baseline", "gnn"),
    ("FAGI", "fagi"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--output", default="experiments/results/exp1.json")
    args = ap.parse_args()
    print("EXPERIMENT 1: CICIDS2017 benchmark (binary; private training set, public reference pool)")
    folds = CICIDSFolds(args.data_path)
    seeds = CONFIG["seeds"]
    per = {k: {"acc": [], "fpr": [], "dr": [], "f1": [], "eps": [], "w01": [], "w05": []} for _, k in MODELS}
    pooled_lab, pooled_pred = [], {k: [] for _, k in MODELS}
    pooled_y, mal_lab, worst01 = [], [], {k: [] for _, k in MODELS}
    for f in range(CONFIG["n_folds"]):
        d = folds.fold(f)
        for s in seeds:
            for _, key in MODELS:
                b = folds.result(key, f, s)
                m = metrics(d["yte"], b["pred_clean"])
                p = per[key]
                p["acc"].append(m["accuracy"]); p["fpr"].append(m["fpr"]); p["dr"].append(m["detection_rate"])
                p["f1"].append(m["f1"]); p["eps"].append(b["info"].get("epsilon"))
                p["w01"].append(float(b["worst_detected"]["0.1"].mean()))
                p["w05"].append(float(b["worst_detected"]["0.5"].mean()))
                if s == seeds[0]:
                    pooled_pred[key].append(b["pred_clean"])
                    worst01[key].append(b["worst_detected"]["0.1"])
            if s == seeds[0]:
                pooled_lab.append(d["labels_te"]); pooled_y.append(d["yte"])
                mal_lab.append(d["labels_te"][d["mal"]])
    lab = np.concatenate(pooled_lab)
    y = np.concatenate(pooled_y)
    rows = []
    for name, key in MODELS:
        p = per[key]
        pred = np.concatenate(pooled_pred[key])
        pc = {}
        for l in sorted(set(lab)):
            msk = lab == l
            want = 0 if l == "BENIGN" else 1
            k_ = int((pred[msk] == want).sum()); n_ = int(msk.sum())
            pc[l] = {"recall": k_ / n_, "k": k_, "n": n_, "ci95": wilson(k_, n_)}
        row = {"name": name, "key": key, "clean": ms(p["acc"]), "fpr": ms(p["fpr"]), "detection_rate": ms(p["dr"]),
               "f1": ms(p["f1"]), "worst_recall_eps0.1": ms(p["w01"]), "worst_recall_eps0.5": ms(p["w05"]),
               "dp": key == "fagi", "epsilon": ms(p["eps"]) if any(e is not None for e in p["eps"]) else None,
               "per_class_seed0": pc}
        rows.append(row)
        print(f"  {name:<20} clean {row['clean']['mean'] * 100:6.2f} ± {row['clean']['std'] * 100:4.2f}  "
              f"FPR {row['fpr']['mean'] * 100:5.2f}  worst recall@0.1 {row['worst_recall_eps0.1']['mean'] * 100:6.2f}"
              f"  @0.5 {row['worst_recall_eps0.5']['mean'] * 100:6.2f}")
    r = {k: per[k] for _, k in MODELS}
    tests = {
        "fagi_vs_gnn_clean": paired(r["fagi"]["acc"], r["gnn"]["acc"]),
        "fagi_vs_gnn_worst0.1": paired(r["fagi"]["w01"], r["gnn"]["w01"]),
        "fagi_vs_gnn_worst0.5": paired(r["fagi"]["w05"], r["gnn"]["w05"]),
        "fagi_vs_knn_worst0.5": paired(r["fagi"]["w05"], r["knn"]["w05"]),
        "rf_vs_fagi_clean": paired(r["rf"]["acc"], r["fagi"]["acc"]),
        "mcnemar_fagi_vs_gnn_clean_seed0": mcnemar(y, np.concatenate(pooled_pred["gnn"]),
                                                   np.concatenate(pooled_pred["fagi"])),
    }
    out = {"config": CONFIG, "config_hash": folds.hash, "env": env_info(),
           "dataset": {k: v for k, v in folds.meta.items() if k != "features"},
           "n_features": len(folds.features), "features": folds.features,
           "split": {"private_train_per_fold": int(len(folds.fold(0)["priv"])),
                     "reference_pool_per_fold": int(len(folds.fold(0)["pool"])),
                     "test_per_fold": int(len(folds.fold(0)["te"])), "attacked_malicious_per_fold": CONFIG["attack_n"]},
           "models": rows, "tests": tests}
    save_json(out, args.output)


if __name__ == "__main__":
    main()
