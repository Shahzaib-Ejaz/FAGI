"""Round-3 robustness checks on seed 0 (5 folds), from cached models.

1. Worst-case attack recall at a matched false-positive rate. Each model's decision threshold is moved so that
   it flags FPR_TARGET of the benign test flows; attacked flows count as detected only if every attack leaves
   their malicious score above that threshold. The attacks are re-scored, not re-optimized: mimicry (all 32
   targets), the model's own stored gradient examples (adaptive PGD for graph models, PGD for flat ones), and
   the two transfer sources. PGD followed by re-retrieval is not stored and is omitted, so these numbers are
   an upper bound on the matched-FPR worst case.
2. Restart curve for 20-step adaptive PGD (1, 2, 5 restarts) at eps_adv = 0.5 on the graph models.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from cicids_folds import CONFIG, FLAT_VARIANTS, GRAPH_VARIANTS, TRANSFER_SOURCES, CICIDSFolds, as_graph, inverse_flow
from cicids_stats import ms, paired
from common import T, env_info, save_json

FPR_TARGET = 0.01
EPS = ["0.1", "0.5"]
MODELS = ["rf", "knn", "mlp_at", "mlp_dp", "gnn", "fagi_no_dp", "fagi", "fagi_sigma0.6", "fagi_sigma2.0", "fagi_sigma4.0"]
RESTART_MODELS = ["gnn", "fagi_no_dp", "fagi"]
RESTARTS = [1, 2, 5]


@torch.no_grad()
def torch_scores(model, X, A, batch=2048):
    model.eval()
    Xt, At = T(X), T(A)
    return torch.cat([F.softmax(model(Xt[i:i + batch], At[i:i + batch]), 1)[:, 1]
                      for i in range(0, len(Xt), batch)]).numpy()


def scorer(folds, name, model, d):
    if name == "knn":
        return lambda Z: d["yr"][d["eb"].neighbours(Z)].mean(axis=1)
    if name == "rf":
        return lambda Z: model.predict_proba(inverse_flow(d["pp"], Z))[:, 1]
    if name in FLAT_VARIANTS:
        return lambda Z: torch_scores(model, *as_graph(Z))
    return lambda Z: torch_scores(model, *d["eb"].build(Z)[:2])


def threshold_for(scores_benign, fpr):
    return float(np.quantile(scores_benign, 1 - fpr, method="higher"))


def fixed_fpr(folds):
    out = {m: {"fpr_default": [], "threshold": [], **{f"default_{e}": [] for e in EPS}, **{f"fixed_{e}": [] for e in EPS}}
           for m in MODELS}
    for f in range(folds.n_splits):
        d = folds.fold(f)
        Zm = d["Zte"][d["mal"]]
        ben = d["Zte"][d["yte"] == 0]
        _, idx = d["benign_nn"].kneighbors(Zm)
        for m in MODELS:
            model = folds.load_model(m, f, 0)
            sc = scorer(folds, m, model, d)
            sb = sc(ben)
            thr_default = 0.5
            thr = threshold_for(sb, FPR_TARGET)
            out[m]["fpr_default"].append(float(np.mean(sb > thr_default)))
            out[m]["threshold"].append(thr)
            blob = folds.result(m, f, 0)
            for e in EPS:
                cand = [sc((Zm + np.clip(d["benign_r"][idx[:, j]] - Zm, -float(e), float(e))).astype(np.float32))
                        for j in range(idx.shape[1])]
                if e in blob["adv_examples"]:
                    cand.append(sc(blob["adv_examples"][e]))
                for src in TRANSFER_SOURCES:
                    if src != m:
                        cand.append(sc(folds.result(src, f, 0)["adv_examples"][e]))
                worst = np.min(np.stack(cand), axis=0)
                out[m][f"default_{e}"].append(float(np.mean(worst > thr_default)))
                out[m][f"fixed_{e}"].append(float(np.mean(worst > thr)))
            print(f"  fold {f} {m:<14} FPR@0.5={out[m]['fpr_default'][-1]:.4f}  "
                  + "  ".join(f"{e}: default {out[m][f'default_{e}'][-1]:.3f} fixed {out[m][f'fixed_{e}'][-1]:.3f}" for e in EPS),
                  flush=True)
    summ = {m: {k: ms(v) for k, v in x.items()} for m, x in out.items()}
    tests = {f"fagi_vs_{o}_fixed_{e}": paired(out["fagi"][f"fixed_{e}"], out[o][f"fixed_{e}"])
             for o in ["gnn", "fagi_no_dp", "knn", "mlp_at"] for e in EPS}
    return {"fpr_target": FPR_TARGET, "models": summ, "tests_seed0": tests}


def restart_curve(folds):
    out = {m: {str(r): [] for r in RESTARTS} for m in RESTART_MODELS}
    for f in range(folds.n_splits):
        d = folds.fold(f)
        Zm = d["Zte"][d["mal"]]
        ym = np.ones(len(Zm), dtype=np.int64)
        for m in RESTART_MODELS:
            model = folds.load_model(m, f, 0)
            pred = folds.predictor(m, model, d)
            for r in RESTARTS:
                za = folds.adaptive_pgd(model, d, Zm, ym, 0.5, CONFIG["pgd_eval_steps"], r, 11000 + 100 * f)
                out[m][str(r)].append(float(np.mean(pred(za) == 1)))
            print(f"  fold {f} {m:<11} " + "  ".join(f"R={r}: {out[m][str(r)][-1]:.3f}" for r in RESTARTS), flush=True)
    return {"eps_adv": 0.5, "steps": CONFIG["pgd_eval_steps"], "curve": {m: {r: ms(v) for r, v in x.items()} for m, x in out.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--output", default="experiments/results/robust_checks.json")
    args = ap.parse_args()
    folds = CICIDSFolds(args.data_path)
    print("[1] worst-case attack recall at a matched 1% FPR (seed 0)")
    a = fixed_fpr(folds)
    print("[2] adaptive PGD restart curve at eps_adv = 0.5 (seed 0)")
    b = restart_curve(folds)
    save_json({"config_hash": folds.hash, "env": env_info(), "fixed_fpr": a, "restart_curve": b}, args.output)


if __name__ == "__main__":
    main()
