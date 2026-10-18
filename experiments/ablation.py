import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cicids_folds import CONFIG, GRAPH_VARIANTS, CICIDSFolds, config_hash
from common import env_info, mean_std, membership_inference, metrics, save_json
from synthetic_common import SYN_CONFIG, SYN_VARIANTS, SyntheticData, load_sessions

CICIDS_ROWS = [
    ("FAGI (full)", "fagi"),
    ("FAGI - GNNGuard", "fagi_no_guard"),
    ("FAGI - PGD", "fagi_no_pgd"),
    ("FAGI - DP-SGD", "fagi_no_dp"),
    ("GNN Baseline", "gnn"),
]
SYN_ROWS = [
    ("FAGI (full)", "fagi"),
    ("FAGI - payload feature", "fagi_no_payload"),
    ("FAGI - GNNGuard", "fagi_no_guard"),
    ("FAGI - PGD", "fagi_no_pgd"),
    ("FAGI - DP-SGD", "fagi_no_dp"),
    ("GNN Baseline (no payload)", "gnn_nopayload"),
]


def cicids(data_path):
    print("\n[Ablation] CICIDS2017")
    folds = CICIDSFolds(data_path)
    res = {k: {"clean": [], "adv": [], "fpr": [], "eps": [], "mia_auc": [], "mia_acc": []} for _, k in CICIDS_ROWS}
    for f in range(CONFIG["n_folds"]):
        d = folds.fold(f)
        for _, k in CICIDS_ROWS:
            model, blob = folds.graph_model(k, f)
            m = metrics(d["yte"], blob["pred_clean"])
            res[k]["clean"].append(m["accuracy"])
            res[k]["fpr"].append(m["fpr"])
            res[k]["adv"].append(float(np.mean(blob["pred_adv"] == d["yte"])))
            res[k]["eps"].append(blob["info"].get("epsilon"))
            ltr, lte = folds.losses(model, f)
            mia = membership_inference(ltr, lte, seed=f)
            res[k]["mia_auc"].append(mia["auc"])
            res[k]["mia_acc"].append(mia["attack_accuracy"])
        print(f"  fold {f + 1}: " + "  ".join(f"{k}={res[k]['clean'][-1]:.4f}/{res[k]['adv'][-1]:.4f}"
                                              for _, k in CICIDS_ROWS), flush=True)
    return rows(CICIDS_ROWS, res, GRAPH_VARIANTS)


def synthetic(api_data):
    print("\n[Ablation] synthetic API sessions")
    D = SyntheticData(load_sessions(api_data))
    idx = np.arange(len(D.y))
    res = {k: {"clean": [], "adv": [], "fpr": [], "eps": []} for _, k in SYN_ROWS}
    for f, (tr, te) in enumerate(D.subset_folds(idx)):
        for _, k in SYN_ROWS:
            model, info, _, Xte = D.train_variant(k, tr, te, seed_offset=1000 * f)
            r = D.evaluate(model, Xte, te, seed=7 + f)
            res[k]["clean"].append(r["clean"])
            res[k]["adv"].append(r["adv"])
            res[k]["fpr"].append(metrics(D.y[te], r["pred_clean"])["fpr"])
            res[k]["eps"].append(info.get("epsilon"))
        print(f"  fold {f + 1}: " + "  ".join(f"{k}={res[k]['clean'][-1]:.3f}/{res[k]['adv'][-1]:.3f}"
                                              for _, k in SYN_ROWS), flush=True)
    return rows(SYN_ROWS, res, SYN_VARIANTS)


def rows(spec, res, variants):
    out = []
    for name, k in spec:
        r = res[k]
        row = {"name": name, "key": k, "clean": mean_std(r["clean"]), "adv": mean_std(r["adv"]),
               "fpr": mean_std(r["fpr"]), "dp": variants[k]["dp"],
               "epsilon": mean_std(r["eps"]) if variants[k]["dp"] else None}
        if "mia_auc" in r:
            row["mia_auc"] = mean_std(r["mia_auc"])
            row["mia_attack_accuracy"] = mean_std(r["mia_acc"])
        out.append(row)
        print(f"  {name:<28} clean {row['clean']['mean'] * 100:6.2f}  adv {row['adv']['mean'] * 100:6.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--api_data", default=None)
    ap.add_argument("--output", default="experiments/results/ablation.json")
    ap.add_argument("--skip_cicids", action="store_true")
    args = ap.parse_args()
    print("ABLATION STUDY")
    out = {"env": env_info(), "cicids_config": CONFIG, "cicids_config_hash": config_hash(),
           "synthetic_config": SYN_CONFIG}
    if not args.skip_cicids:
        out["cicids2017"] = cicids(args.data_path)
    out["synthetic"] = synthetic(args.api_data)
    save_json(out, args.output)


if __name__ == "__main__":
    main()
