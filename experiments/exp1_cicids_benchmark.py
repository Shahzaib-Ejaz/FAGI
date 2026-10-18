import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cicids_folds import CONFIG, CICIDSFolds, config_hash
from common import env_info, mcnemar, mean_std, metrics, per_class_detection, save_json

MODELS = [
    ("Random Forest", "rf", "flat"),
    ("MLP", "mlp", "flat"),
    ("1D-CNN", "cnn", "flat"),
    ("LSTM", "lstm", "flat"),
    ("GNN Baseline", "gnn", "graph"),
    ("FAGI", "fagi", "graph"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--output", default="experiments/results/exp1.json")
    args = ap.parse_args()

    print("EXPERIMENT 1: CICIDS2017 benchmark (binary benign vs attack)")
    folds = CICIDSFolds(args.data_path)
    print(f"  sample: {len(folds.y)} flows, {len(folds.features)} features, config {config_hash()}")

    per = {key: {"acc": [], "fpr": [], "dr": [], "f1": [], "adv": [], "eps": []} for _, key, _ in MODELS}
    pooled = {key: {"clean": [], "adv": []} for _, key, _ in MODELS}
    y_all, lab_all = [], []

    for f in range(CONFIG["n_folds"]):
        print(f"  fold {f + 1}/{CONFIG['n_folds']}", flush=True)
        d = folds.fold(f)
        y_all.append(d["yte"])
        lab_all.append(d["labels_te"])
        for _, key, kind in MODELS:
            _, blob = folds.flat_model(key, f) if kind == "flat" else folds.graph_model(key, f)
            m = metrics(d["yte"], blob["pred_clean"])
            per[key]["acc"].append(m["accuracy"])
            per[key]["fpr"].append(m["fpr"])
            per[key]["dr"].append(m["detection_rate"])
            per[key]["f1"].append(m["f1"])
            per[key]["eps"].append(blob["info"].get("epsilon"))
            pooled[key]["clean"].append(blob["pred_clean"])
            if blob["pred_adv"] is not None:
                per[key]["adv"].append(float(np.mean(blob["pred_adv"] == d["yte"])))
                pooled[key]["adv"].append(blob["pred_adv"])

    y_all = np.concatenate(y_all)
    lab_all = np.concatenate(lab_all)
    rows = []
    for name, key, kind in MODELS:
        p = per[key]
        clean_pool = np.concatenate(pooled[key]["clean"])
        row = {"name": name, "key": key,
               "clean": mean_std(p["acc"]), "fpr": mean_std(p["fpr"]),
               "detection_rate": mean_std(p["dr"]), "f1": mean_std(p["f1"]),
               "adv": mean_std(p["adv"]) if p["adv"] else None,
               "dp": key == "fagi",
               "epsilon": mean_std(p["eps"]) if any(e is not None for e in p["eps"]) else None,
               "per_class_detection": per_class_detection(lab_all, clean_pool)}
        if pooled[key]["adv"]:
            row["per_class_detection_adv"] = per_class_detection(lab_all, np.concatenate(pooled[key]["adv"]))
        rows.append(row)
        adv = f"{row['adv']['mean'] * 100:6.2f}" if row["adv"] else "   n/a"
        print(f"  {name:<14} clean {row['clean']['mean'] * 100:6.2f} ± {row['clean']['std'] * 100:4.2f}  "
              f"adv {adv}  FPR {row['fpr']['mean'] * 100:5.2f}")

    g_clean, f_clean = np.concatenate(pooled["gnn"]["clean"]), np.concatenate(pooled["fagi"]["clean"])
    g_adv, f_adv = np.concatenate(pooled["gnn"]["adv"]), np.concatenate(pooled["fagi"]["adv"])
    out = {
        "config": CONFIG, "config_hash": config_hash(), "env": env_info(),
        "dataset": {k: v for k, v in folds.meta.items() if k != "features"},
        "n_features": len(folds.features), "features": folds.features,
        "models": rows,
        "mcnemar_fagi_vs_gnn_clean": mcnemar(y_all, g_clean, f_clean),
        "mcnemar_fagi_vs_gnn_adv": mcnemar(y_all, g_adv, f_adv),
    }
    save_json(out, args.output)


if __name__ == "__main__":
    main()
