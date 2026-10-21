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
import json
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
FPR_GRID = [0.005, 0.01, 0.02]
EPS = [str(e) for e in CONFIG["eps_sweep"]]
TEST_EPS = ["0.1", "0.5"]
MODELS = ["rf", "knn", "mlp_at", "mlp_dp", "gnn", "fagi_no_dp", "fagi", "fagi_no_guard", "fagi_no_pgd",
          "fagi_sigma0.6", "fagi_sigma2.0", "fagi_sigma4.0"]
TEMPORAL_MODELS = ["rf", "knn", "mlp", "mlp_at", "mlp_dp", "gnn", "fagi_no_dp", "fagi"]
MAIN_MODELS = ["rf", "knn", "mlp", "mlp_at", "mlp_dp", "cnn", "gnn", "fagi_no_dp", "fagi"]
RESTART_MODELS = ["gnn", "fagi_no_dp", "fagi"]
RESTARTS = [1, 2, 5]


@torch.no_grad()
def torch_scores(model, X, A, batch=2048):
    model.eval()
    Xt, At = T(X), T(A)
    return torch.cat([F.softmax(model(Xt[i:i + batch], At[i:i + batch]), 1)[:, 1]
                      for i in range(0, len(Xt), batch)]).numpy()


def scorer(folds, name, model, d):
    """Malicious score in [0, 1]; score > 0.5 reproduces each model's deployed prediction."""
    if name == "knn":
        def knn_score(Z, self_index=None):
            nbr = d["eb"].neighbours(Z, self_index)
            votes = d["yr"][nbr].mean(axis=1)
            return votes + 1e-6 * (d["yr"][nbr[:, 0]] - 0.5)
        return knn_score
    if name == "rf":
        return lambda Z, self_index=None: model.predict_proba(inverse_flow(d["pp"], Z))[:, 1]
    if name in FLAT_VARIANTS:
        return lambda Z, self_index=None: torch_scores(model, *as_graph(Z))
    return lambda Z, self_index=None: torch_scores(model, *d["eb"].build(Z, self_index=self_index)[:2])


def scores_for(folds, m, f, s_, d, idx, cache_dir):
    """Benign test scores, benign pool scores (self excluded), and per-flow worst attacked scores; cached."""
    path = os.path.join(cache_dir, f"scores-{folds.protocol}_{m}_f{f}_s{s_}_{folds.hash}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    model = folds.load_model(m, f, s_)
    sc = scorer(folds, m, model, d)
    Zm = d["Zte"][d["mal"]]
    pool_ben = np.flatnonzero(d["yr"] == 0)
    out = {"benign_test": sc(d["Zte"][d["yte"] == 0]), "benign_pool": sc(d["Zr"][pool_ben], self_index=pool_ben)}
    blob = folds.result(m, f, s_)
    for e in EPS:
        cand = [sc((Zm + np.clip(d["benign_r"][idx[:, j]] - Zm, -float(e), float(e))).astype(np.float32))
                for j in range(idx.shape[1])]
        if e in blob["adv_examples"]:
            cand.append(sc(blob["adv_examples"][e]))
        for src in TRANSFER_SOURCES:
            if src != m:
                cand.append(sc(folds.result(src, f, s_)["adv_examples"][e]))
        out[f"worst_{e}"] = np.min(np.stack(cand), axis=0)
    np.savez_compressed(path, **out)
    return out


def jitter(x, rng):
    """Break ties at random so discrete scorers (k-NN votes, forest probabilities) reach the target FPR."""
    return x + rng.uniform(0, 1e-7, size=len(x))


def evaluate(sc, rng):
    bt, bp = jitter(sc["benign_test"], rng), jitter(sc["benign_pool"], rng)
    worst = {e: jitter(sc[f"worst_{e}"], rng) for e in EPS}
    row = {"fpr_default": float(np.mean(sc["benign_test"] > 0.5))}
    for e in EPS:
        row[f"default_{e}"] = float(np.mean(sc[f"worst_{e}"] > 0.5))
    for fpr in FPR_GRID:
        thr = float(np.quantile(bt, 1 - fpr))
        tag = f"{fpr:g}"
        row[f"realized_fpr_{tag}"] = float(np.mean(bt > thr))
        row[f"threshold_{tag}"] = thr
        for e in EPS:
            row[f"fixed_{tag}_{e}"] = float(np.mean(worst[e] > thr))
    thr = float(np.quantile(bp, 1 - FPR_TARGET))
    row["pool_threshold"] = thr
    row["pool_realized_test_fpr"] = float(np.mean(bt > thr))
    for e in EPS:
        row[f"pool_{e}"] = float(np.mean(worst[e] > thr))
    return row


def fixed_fpr(folds, models=MODELS, seeds=(0,)):
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "cache")
    out = {m: {} for m in models}
    for f, s_ in [(f, s_) for f in range(folds.n_splits) for s_ in seeds]:
        d = folds.fold(f)
        _, idx = d["benign_nn"].kneighbors(d["Zte"][d["mal"]])
        for m in models:
            row = evaluate(scores_for(folds, m, f, s_, d, idx, cache_dir), np.random.RandomState(12000 + 100 * f + s_))
            for k_, v_ in row.items():
                out[m].setdefault(k_, []).append(v_)
            print(f"  [{folds.protocol}] fold {f} seed {s_} {m:<14} FPR@0.5={row['fpr_default']:.4f}  "
                  + "  ".join(f"{e}: default {row[f'default_{e}']:.3f} 1% {row[f'fixed_0.01_{e}']:.3f} pool {row[f'pool_{e}']:.3f}"
                              for e in TEST_EPS), flush=True)
    ns = len(seeds)

    def fm(v):
        return np.asarray(v, float).reshape(folds.n_splits, ns).mean(axis=1).tolist()

    summ = {m: {k: ms(fm(v) if m == "knn" else v) for k, v in x.items()} for m, x in out.items()}
    for m in summ:
        for e in EPS:
            summ[m][f"fixed_{e}"] = summ[m][f"fixed_{FPR_TARGET:g}_{e}"]
    tests = {}

    def cmp(a, b, key):
        if folds.n_splits * (1 if "knn" in (a, b) else ns) < 3:
            d_ = np.asarray(out[a][key]) - np.asarray(out[b][key])
            return {"mean_diff": float(d_.mean()), "p": None, "n": int(len(d_)), "wins_a": int((d_ > 0).sum()),
                    "wins_b": int((d_ < 0).sum()), "unit": "too few units for a test"}
        if "knn" in (a, b):
            r = paired(fm(out[a][key]), fm(out[b][key])); r["unit"] = "fold (seeds averaged)"
        else:
            r = paired(out[a][key], out[b][key]); r["unit"] = "fold-seed pair" if ns > 1 else "fold"
        return r

    if "fagi" in out:
        for o in ["gnn", "fagi_no_dp", "knn", "mlp_at", "mlp_dp"]:
            if o in out:
                for e in TEST_EPS:
                    tests[f"fagi_vs_{o}_fixed_{e}"] = cmp("fagi", o, f"fixed_{FPR_TARGET:g}_{e}")
                    tests[f"fagi_vs_{o}_pool_{e}"] = cmp("fagi", o, f"pool_{e}")
                    for fpr in FPR_GRID:
                        tests[f"fagi_vs_{o}_fixed_{fpr:g}_{e}"] = cmp("fagi", o, f"fixed_{fpr:g}_{e}")
    for v in ["fagi_no_guard", "fagi_no_pgd"]:
        if v in out and "fagi" in out:
            for e in TEST_EPS:
                tests[f"fagi_vs_{v}_fixed_{e}"] = cmp("fagi", v, f"fixed_{FPR_TARGET:g}_{e}")
                tests[f"fagi_vs_{v}_pool_{e}"] = cmp("fagi", v, f"pool_{e}")
    if "fagi_no_dp" in out and "gnn" in out:
        for e in TEST_EPS:
            tests[f"fagi_no_dp_vs_gnn_fixed_{e}"] = cmp("fagi_no_dp", "gnn", f"fixed_{FPR_TARGET:g}_{e}")
            tests[f"fagi_no_dp_vs_gnn_pool_{e}"] = cmp("fagi_no_dp", "gnn", f"pool_{e}")
    return {"fpr_target": FPR_TARGET, "fpr_grid": FPR_GRID, "seeds": list(seeds), "models": summ, "tests": tests,
            "calibration": "threshold at the (1 - FPR) quantile of benign test scores (oracle, same for every model), "
                           "or of benign reference-pool scores with each flow excluded from its own neighborhood (pool_*); "
                           "ties broken at random"}


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
    ap.add_argument("--skip_restarts", action="store_true", help="reuse restart_curve from an existing output")
    args = ap.parse_args()
    folds = CICIDSFolds(args.data_path)
    print("[1] matched-FPR worst-case recall, seed 0, all variants")
    a = fixed_fpr(folds)
    print("[1b] matched-FPR worst-case recall, all seeds, main models")
    a3 = fixed_fpr(folds, MAIN_MODELS, CONFIG["seeds"])
    print("[1c] matched-FPR worst-case recall, temporal split")
    at = fixed_fpr(CICIDSFolds(args.data_path, protocol="temporal"), TEMPORAL_MODELS, CONFIG["seeds"])
    if args.skip_restarts and os.path.exists(args.output):
        with open(args.output) as fh:
            b = json.load(fh)["restart_curve"]
    else:
        print("[2] adaptive PGD restart curve at eps_adv = 0.5 (seed 0)")
        b = restart_curve(folds)
    save_json({"config_hash": folds.hash, "env": env_info(), "fixed_fpr_seed0": a, "fixed_fpr": a3, "fixed_fpr_temporal": at,
               "restart_curve": b},
              args.output)


if __name__ == "__main__":
    main()
