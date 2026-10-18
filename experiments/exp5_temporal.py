import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cicids_folds import CONFIG, CICIDSFolds
from cicids_stats import ms, wilson
from common import env_info, metrics, save_json

MODELS = [("Random Forest", "rf"), ("k-NN vote (k=8)", "knn"), ("MLP", "mlp"), ("MLP + PGD training", "mlp_at"),
          ("MLP + PGD + DP-SGD, all data private", "mlp_dp"), ("GNN Baseline", "gnn"), ("FAGI - DP-SGD", "fagi_no_dp"),
          ("FAGI", "fagi")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--output", default="experiments/results/exp5_temporal.json")
    args = ap.parse_args()
    print("EXPERIMENT 5: temporal split (earlier 70% of each day file trains, later 30% tests)")
    folds = CICIDSFolds(args.data_path, protocol="temporal")
    d = folds.fold(0)
    lab_tr = folds.labels[d["tr"]]
    lab_te = d["labels_te"]
    counts = {l: {"train": int((lab_tr == l).sum()), "test": int((lab_te == l).sum())}
              for l in sorted(set(folds.labels))}
    unseen = [l for l, c in counts.items() if c["train"] == 0 and c["test"] > 0]
    rows = []
    for name, key in MODELS:
        acc, fpr, dr, w01, w05, eps = [], [], [], [], [], []
        pred0 = None
        for s in CONFIG["seeds"]:
            b = folds.result(key, 0, s)
            w = folds.worst_detected(key, 0, s)
            m = metrics(d["yte"], b["pred_clean"])
            acc.append(m["accuracy"]); fpr.append(m["fpr"]); dr.append(m["detection_rate"])
            w01.append(float(w["0.1"].mean())); w05.append(float(w["0.5"].mean())); eps.append(b["info"].get("epsilon"))
            if s == CONFIG["seeds"][0]:
                pred0 = b["pred_clean"]
        pc = {}
        for l in sorted(set(lab_te)):
            msk = lab_te == l
            want = 0 if l == "BENIGN" else 1
            k_, n_ = int((pred0[msk] == want).sum()), int(msk.sum())
            pc[l] = {"recall": k_ / n_, "k": k_, "n": n_, "ci95": wilson(k_, n_)}
        det = key == "knn"
        row = {"name": name, "key": key, "deterministic_across_seeds": det,
               "clean": ms(acc[:1] if det else acc), "fpr": ms(fpr[:1] if det else fpr),
               "detection_rate": ms(dr[:1] if det else dr), "worst_recall_eps0.1": ms(w01[:1] if det else w01),
               "worst_recall_eps0.5": ms(w05[:1] if det else w05),
               "epsilon": ms(eps) if any(e is not None for e in eps) else None, "per_class_seed0": pc}
        rows.append(row)
        print(f"  {name:<40} clean {row['clean']['mean'] * 100:6.2f}  DR {row['detection_rate']['mean'] * 100:6.2f}  "
              f"FPR {row['fpr']['mean'] * 100:5.2f}  worst@0.1 {row['worst_recall_eps0.1']['mean'] * 100:6.2f}  "
              f"@0.5 {row['worst_recall_eps0.5']['mean'] * 100:6.2f}")
    save_json({"config": CONFIG, "config_hash": folds.hash, "env": env_info(), "protocol": "temporal",
               "split": {"train": int(len(d["tr"])), "private_train": int(len(d["priv"])),
                         "reference_pool": int(len(d["pool"])), "test": int(len(d["te"])),
                         "attacked_malicious": int(len(d["mal"]))},
               "label_counts": counts, "labels_only_in_test": unseen, "models": rows}, args.output)


if __name__ == "__main__":
    main()
