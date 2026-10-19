import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

from cicids_folds import CONFIG, GRAPH_VARIANTS, CICIDSFolds
from cicids_stats import ms
from common import DenseFAGI, env_info, per_sample_loss, save_json, train_model

N_CANARY = 200
MODELS = ["gnn", "fagi_no_dp", "fagi"]


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def empirical_eps_lower_bound(loss_in, loss_out, delta):
    s_in, s_out = -np.asarray(loss_in), -np.asarray(loss_out)
    n_in, n_out = len(s_in), len(s_out)
    best = 0.0
    for t in np.unique(np.concatenate([s_in, s_out])):
        tp, fp = int((s_in >= t).sum()), int((s_out >= t).sum())
        tpr_lo, _ = clopper_pearson(tp, n_in)
        _, fpr_hi = clopper_pearson(fp, n_out)
        if fpr_hi > 0 and tpr_lo - delta > 0:
            best = max(best, float(np.log((tpr_lo - delta) / fpr_hi)))
        tnr_lo, _ = clopper_pearson(n_out - fp, n_out)
        _, fnr_hi = clopper_pearson(n_in - tp, n_in)
        if fnr_hi > 0 and tnr_lo - delta > 0:
            best = max(best, float(np.log((tnr_lo - delta) / fnr_hi)))
    return best


def bootstrap_auc(loss_in, loss_out, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    li, lo = np.asarray(loss_in), np.asarray(loss_out)
    vals = []
    for _ in range(n_boot):
        a = rng.choice(li, len(li), replace=True)
        b = rng.choice(lo, len(lo), replace=True)
        vals.append(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], -np.r_[a, b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_fold(folds, f):
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cache", f"canary_f{f}_{folds.hash}.json")
    if os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh)
    d = folds.fold(f)
    rng = np.random.RandomState(4000 + f)
    can = rng.choice(len(d["yp"]), N_CANARY, replace=False)
    in_mask = np.zeros(N_CANARY, dtype=bool)
    in_mask[rng.choice(N_CANARY, N_CANARY // 2, replace=False)] = True
    can_in, can_out = can[in_mask], can[~in_mask]
    y_flip = d["yp"].copy()
    y_flip[can] = 1 - y_flip[can]
    keep = np.setdiff1d(np.arange(len(y_flip)), can_out)
    out = {}
    for name in MODELS:
        v = GRAPH_VARIANTS[name]
        model = DenseFAGI(d["Gp"].shape[2], gnnguard_threshold=CONFIG["gnnguard_tau"] if v["guard"] else None,
                          readout="center")
        info = train_model(model, d["Gp"][keep], d["Ap"][keep], y_flip[keep], node_mask=d["Mp"][keep],
                           epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"], lr=CONFIG["lr"], adv=v["adv"],
                           dp=v["dp"], sigma=CONFIG["dp_sigma"], clip=CONFIG["dp_clip"], delta=CONFIG["dp_delta"],
                           seed=5000 + 100 * f + v["seed"], pgd_eps=CONFIG["pgd_eps"],
                           pgd_steps=CONFIG["pgd_train_steps"])
        loss = per_sample_loss(model, d["Gp"][can], d["Ap"][can], y_flip[can])
        auc = float(roc_auc_score(in_mask.astype(int), -loss))
        fit_in = float(np.mean(loss[in_mask] < np.log(2)))
        fit_out = float(np.mean(loss[~in_mask] < np.log(2)))
        out[name] = {"auc": auc, "canary_fit_rate_in": fit_in, "canary_fit_rate_out": fit_out,
                     "loss_in": loss[in_mask].tolist(), "loss_out": loss[~in_mask].tolist(),
                     "epsilon": info.get("epsilon")}
        print(f"  fold {f} {name:<11} canary AUC={auc:.3f}  fit in={fit_in:.2f} out={fit_out:.2f}  eps={info.get('epsilon')}",
              flush=True)
    with open(cache, "w") as fh:
        json.dump(out, fh)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--output", default="experiments/results/mia_canary.json")
    args = ap.parse_args()
    print(f"CANARY AUDIT: {N_CANARY} label-flipped private flows per fold, half included in training")
    folds = CICIDSFolds(args.data_path)
    per_fold = [run_fold(folds, f) for f in range(CONFIG["n_folds"])]
    summary = {}
    for name in MODELS:
        li = np.concatenate([p[name]["loss_in"] for p in per_fold])
        lo = np.concatenate([p[name]["loss_out"] for p in per_fold])
        pooled = float(roc_auc_score(np.r_[np.ones(len(li)), np.zeros(len(lo))], -np.r_[li, lo]))
        summary[name] = {"auc_pooled": pooled, "auc_pooled_ci95": bootstrap_auc(li, lo),
                         "empirical_eps_lower_bound": empirical_eps_lower_bound(li, lo, CONFIG["dp_delta"]),
                         "auc_per_fold": ms([p[name]["auc"] for p in per_fold]),
                         "fit_rate_in": ms([p[name]["canary_fit_rate_in"] for p in per_fold]),
                         "fit_rate_out": ms([p[name]["canary_fit_rate_out"] for p in per_fold]),
                         "epsilon": per_fold[0][name]["epsilon"], "n_in": int(len(li)), "n_out": int(len(lo))}
        print(f"  {name:<11} pooled AUC={pooled:.3f} CI={summary[name]['auc_pooled_ci95']}  eps_lb={summary[name]['empirical_eps_lower_bound']:.3f}  in-fit={summary[name]['fit_rate_in']['mean']:.2f}  "
              f"out-fit={summary[name]['fit_rate_out']['mean']:.2f}")
    save_json({"config": CONFIG, "config_hash": folds.hash, "env": env_info(), "n_canary_per_fold": N_CANARY,
               "summary": summary}, args.output)


if __name__ == "__main__":
    main()
