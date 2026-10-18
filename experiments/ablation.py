import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cicids_folds import CONFIG, GRAPH_VARIANTS, CICIDSFolds
from cicids_stats import ms, paired
from common import env_info, metrics, save_json
from mia import mia_report
from synthetic_common import SYN_CONFIG, SYN_VARIANTS, SyntheticData, load_sessions

CICIDS_ROWS = [("FAGI (full)", "fagi"), ("FAGI - GNNGuard", "fagi_no_guard"), ("FAGI - PGD", "fagi_no_pgd"),
               ("FAGI - DP-SGD", "fagi_no_dp"), ("GNN Baseline", "gnn")]
SYN_ROWS = [("FAGI (full)", "fagi"), ("FAGI - payload feature", "fagi_no_payload"), ("FAGI - GNNGuard", "fagi_no_guard"),
            ("FAGI - PGD", "fagi_no_pgd"), ("FAGI - DP-SGD", "fagi_no_dp"), ("GNN Baseline (no payload)", "gnn_nopayload")]
MIA_MODELS = ["gnn_overfit", "gnn", "fagi_no_dp", "fagi"]


def cicids(data_path):
    print("\n[Ablation] CICIDS2017")
    folds = CICIDSFolds(data_path)
    res = {k: {"clean": [], "fpr": [], "w01": [], "w05": [], "eps": []} for _, k in CICIDS_ROWS}
    sig = {s: {"clean": [], "w01": [], "w05": [], "eps": []} for s in ["fagi_no_dp"] + [f"fagi_sigma{x}" for x in CONFIG["sigma_sweep"]] + ["fagi"]}
    mia = {k: [] for k in MIA_MODELS}
    for f in range(CONFIG["n_folds"]):
        d = folds.fold(f)
        for s in [0]:
            for _, k in CICIDS_ROWS:
                b = folds.result(k, f, s)
                w = folds.worst_detected(k, f, s)
                m = metrics(d["yte"], b["pred_clean"])
                r = res[k]
                r["clean"].append(m["accuracy"]); r["fpr"].append(m["fpr"]); r["eps"].append(b["info"].get("epsilon"))
                r["w01"].append(float(w["0.1"].mean()))
                r["w05"].append(float(w["0.5"].mean()))
        for k in sig:
            b = folds.result(k, f, 0)
            w = folds.worst_detected(k, f, 0)
            sig[k]["clean"].append(float(np.mean(b["pred_clean"] == d["yte"])))
            sig[k]["w01"].append(float(w["0.1"].mean()))
            sig[k]["w05"].append(float(w["0.5"].mean()))
            sig[k]["eps"].append(b["info"].get("epsilon"))
        for k in MIA_MODELS:
            b = folds.result(k, f, 0)
            mm = b["mia"]
            rep = mia_report(mm["loss_members"], mm["loss_nonmembers"], mm["labels_members"], d["labels_te"], seed=f)
            rep["median_loss_members"] = float(np.median(mm["loss_members"]))
            rep["median_loss_nonmembers"] = float(np.median(mm["loss_nonmembers"]))
            rep["mean_loss_members"] = float(np.mean(mm["loss_members"]))
            rep["mean_loss_nonmembers"] = float(np.mean(mm["loss_nonmembers"]))
            mia[k].append(rep)
        print(f"  fold {f + 1}: " + "  ".join(f"{k}={res[k]['clean'][-1]:.4f}/{res[k]['w05'][-1]:.3f}"
                                              for _, k in CICIDS_ROWS), flush=True)
    rows = []
    for name, k in CICIDS_ROWS:
        r = res[k]
        rows.append({"name": name, "key": k, "clean": ms(r["clean"]), "fpr": ms(r["fpr"]),
                     "worst_recall_eps0.1": ms(r["w01"]), "worst_recall_eps0.5": ms(r["w05"]),
                     "dp": GRAPH_VARIANTS[k]["dp"], "epsilon": ms(r["eps"]) if GRAPH_VARIANTS[k]["dp"] else None})
    sweep = []
    for k, r in sig.items():
        sigma = None if k == "fagi_no_dp" else GRAPH_VARIANTS[k].get("sigma", CONFIG["dp_sigma"])
        sweep.append({"key": k, "sigma": sigma, "epsilon": ms(r["eps"]) if sigma else None,
                      "clean": ms(r["clean"]), "worst_recall_eps0.1": ms(r["w01"]), "worst_recall_eps0.5": ms(r["w05"])})
    sweep.sort(key=lambda x: -1 if x["sigma"] is None else x["sigma"])
    mia_out = {}
    for k, lst in mia.items():
        mia_out[k] = {f: ms([x[f] for x in lst]) for f in ["auc", "attack_accuracy", "tpr_at_1pct_fpr",
                                                            "tpr_at_0.1pct_fpr", "max_per_class_auc",
                                                            "median_loss_members", "median_loss_nonmembers",
                                                            "mean_loss_members", "mean_loss_nonmembers"]}
    tests = {"fagi_vs_no_dp_clean": paired(res["fagi"]["clean"], res["fagi_no_dp"]["clean"]),
             "fagi_vs_no_pgd_w05": paired(res["fagi"]["w05"], res["fagi_no_pgd"]["w05"]),
             "fagi_vs_no_guard_w05": paired(res["fagi"]["w05"], res["fagi_no_guard"]["w05"]),
             "fagi_vs_no_guard_clean": paired(res["fagi"]["clean"], res["fagi_no_guard"]["clean"])}
    return {"rows_seed0": rows, "sigma_sweep_seed0": sweep, "mia_seed0": mia_out, "tests_seed0": tests}


def synthetic(api_data):
    print("\n[Ablation] synthetic API sessions")
    D = SyntheticData(load_sessions(api_data))
    idx = np.arange(len(D.y))
    res = {k: {"clean": [], "adv": [], "fpr": [], "eps": []} for _, k in SYN_ROWS}
    for f, (tr, te) in enumerate(D.subset_folds(idx)):
        for s in SYN_CONFIG["seeds"]:
            for _, k in SYN_ROWS:
                model, info, _, Xte = D.train_variant(k, tr, te, seed_offset=1000 * f + 100 * s)
                r = D.evaluate(model, Xte, te, seed=7 + 10 * f + s)
                res[k]["clean"].append(r["clean"]); res[k]["adv"].append(r["adv"])
                res[k]["fpr"].append(metrics(D.y[te], r["pred_clean"])["fpr"]); res[k]["eps"].append(info.get("epsilon"))
        print(f"  fold {f + 1}: " + "  ".join(f"{k}={np.mean(res[k]['clean'][-len(SYN_CONFIG['seeds']):]):.3f}/{np.mean(res[k]['adv'][-len(SYN_CONFIG['seeds']):]):.3f}"
                                              for _, k in SYN_ROWS), flush=True)
    return [{"name": n, "key": k, "clean": ms(res[k]["clean"]), "adv": ms(res[k]["adv"]), "fpr": ms(res[k]["fpr"]),
             "dp": SYN_VARIANTS[k]["dp"], "epsilon": ms(res[k]["eps"]) if SYN_VARIANTS[k]["dp"] else None}
            for n, k in SYN_ROWS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--api_data", default=None)
    ap.add_argument("--output", default="experiments/results/ablation.json")
    ap.add_argument("--skip_cicids", action="store_true")
    ap.add_argument("--skip_synthetic", action="store_true")
    args = ap.parse_args()
    print("ABLATION STUDY")
    out = {"env": env_info(), "cicids_config": CONFIG, "synthetic_config": SYN_CONFIG}
    if not args.skip_cicids:
        out["cicids2017"] = cicids(args.data_path)
    if not args.skip_synthetic:
        out["synthetic"] = synthetic(args.api_data)
    save_json(out, args.output)


if __name__ == "__main__":
    main()
