"""Robustness checks on the cached CICIDS2017 models.

1. Worst-case attack recall at a matched false-positive rate. Each model's decision threshold is moved so that
   it flags FPR_TARGET of the benign test flows; an attacked flow counts as detected only if the lowest
   malicious score that any attack reached for it stays above that threshold. The scores come from
   strong_attacks.py, which minimizes each model's score (so the attacks are optimized for every threshold at
   once) and returns the deployed-path score of each flow's best example. rescored_* keys repeat the analysis
   with the stored attacks only (mimicry, the model's stored gradient examples, the two stored transfer
   sources), which stop at the default decision boundary and were merely re-scored at the new threshold.
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
from cicids_stats import ms, paired_folds
from common import env_info, save_json
import strong_attacks as sa
from strong_attacks import scorer  # noqa: F401  (deployed-path scorer, re-exported)

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


def rescored(full):
    """Lowest score over the stored attacks that the earlier matched-FPR analysis re-scored."""
    return np.min(np.stack([full[k] for k in sa.OLD_FAMILIES if k in full]), axis=0)


def worst_detected(folds, name, f, s, eps_list=None):
    """Per-flow detection at the default threshold: the stored attacks and the full suite must all fail."""
    old = folds.worst_detected(name, f, s)
    return {e: old[e] & (sa.suite(folds, name, f, s, e)["final"] > 0.5) for e in (eps_list or sa.EPS)}


FAMILIES = ("neighborhood", "score_attack", "transfer_strong", "random_search")


def scores_for(folds, m, f, s_, eps_list):
    """Benign test and pool scores, and per-flow lowest attacked scores (full suite and stored attacks only)."""
    base = sa.base_scores(folds, m, f, s_)
    out = {"benign_test": base["benign_test"], "benign_pool": base["benign_pool"], "discrete": m in ("knn", "rf"),
           "floor": sa.floor_of(m)}
    for e in eps_list:
        full = sa.suite(folds, m, f, s_, e)
        out[f"worst_{e}"] = full["final"]
        out[f"rescored_{e}"] = rescored(full)
        if m == "rf":
            out[f"rs_curve_{e}"] = full["random_search_curve"]
        for k_ in FAMILIES:
            if k_ in full:
                out[f"{k_}_{e}"] = full[k_]
    return out


def jitter(x, rng, scale=1e-7):
    """Break ties at random so discrete scorers (k-NN votes, forest probabilities) reach the target FPR.

    Neural scores are float64 and essentially tie-free; they get a jitter far below their spacing."""
    return x + rng.uniform(0, scale, size=len(x))


def evaluate(sc, rng, eps_list):
    js = 1e-7 if sc["discrete"] else 1e-15
    bt, bp = jitter(sc["benign_test"], rng, js), jitter(sc["benign_pool"], rng, js)
    worst = {e: jitter(sc[f"worst_{e}"], rng, js) for e in eps_list}
    old = {e: jitter(sc[f"rescored_{e}"], rng, js) for e in eps_list}
    row = {"fpr_default": float(np.mean(sc["benign_test"] > 0.5))}
    for e in eps_list:
        row[f"default_{e}"] = float(np.mean(sc[f"worst_{e}"] > 0.5))
        row[f"rescored_default_{e}"] = float(np.mean(sc[f"rescored_{e}"] > 0.5))
        for k_ in FAMILIES:
            if f"{k_}_{e}" in sc:
                row[f"{k_}_default_{e}"] = float(np.mean(sc[f"{k_}_{e}"] > 0.5))
        if f"rs_curve_{e}" in sc:
            for q, v in zip(sa.SUITE["rs_checkpoints"], sc[f"rs_curve_{e}"]):
                row[f"rs_curve_{q}_{e}"] = float(v)
    lowest = 1.0
    for fpr in FPR_GRID:
        thr = float(np.quantile(bt, 1 - fpr))
        lowest = min(lowest, thr)
        tag = f"{fpr:g}"
        row[f"realized_fpr_{tag}"] = float(np.mean(bt > thr))
        row[f"threshold_{tag}"] = thr
        for e in eps_list:
            row[f"fixed_{tag}_{e}"] = float(np.mean(worst[e] > thr))
            row[f"rescored_fixed_{tag}_{e}"] = float(np.mean(old[e] > thr))
    thr = float(np.quantile(bp, 1 - FPR_TARGET))
    lowest = min(lowest, thr)
    row["pool_threshold"] = thr
    row["pool_realized_test_fpr"] = float(np.mean(bt > thr))
    for e in eps_list:
        row[f"pool_{e}"] = float(np.mean(worst[e] > thr))
    # The attacks stop lowering a flow's score once it is under score_floor, so every threshold must sit above it.
    row["threshold_above_score_floor"] = float(lowest > sc["floor"])
    return row


def fixed_fpr(folds, models=MODELS, seeds=(0,), eps_list=EPS):
    out = {m: {} for m in models}
    for f, s_ in [(f, s_) for f in range(folds.n_splits) for s_ in seeds]:
        for m in models:
            row = evaluate(scores_for(folds, m, f, s_, eps_list), np.random.RandomState(12000 + 100 * f + s_), eps_list)
            for k_, v_ in row.items():
                out[m].setdefault(k_, []).append(v_)
            print(f"  [{folds.protocol}] fold {f} seed {s_} {m:<14} FPR@0.5={row['fpr_default']:.4f}  "
                  + "  ".join(f"{e}: default {row[f'default_{e}']:.3f} 1% {row[f'fixed_0.01_{e}']:.3f} (stored only {row[f'rescored_fixed_0.01_{e}']:.3f}) pool {row[f'pool_{e}']:.3f}"
                              for e in TEST_EPS), flush=True)
    ns = len(seeds)

    def fm(v):
        return np.asarray(v, float).reshape(folds.n_splits, ns).mean(axis=1).tolist()

    summ = {m: {k: ms(fm(v) if m == "knn" else v) for k, v in x.items()} for m, x in out.items()}
    for m in summ:
        assert min(summ[m]["threshold_above_score_floor"]["values"]) == 1.0, f"{m}: a threshold is below the attack score floor"
        for e in eps_list:
            summ[m][f"fixed_{e}"] = summ[m][f"fixed_{FPR_TARGET:g}_{e}"]
            summ[m][f"rescored_fixed_{e}"] = summ[m][f"rescored_fixed_{FPR_TARGET:g}_{e}"]
    tests = {}

    def cmp(a, b, key, key_b=None):
        key_b = key_b or key
        if folds.n_splits < 3:
            d_ = np.asarray(out[a][key]) - np.asarray(out[b][key_b])
            return {"mean_diff": float(d_.mean()), "p": None, "n": int(len(d_)), "wins_a": int((d_ > 0).sum()),
                    "wins_b": int((d_ < 0).sum()), "unit": "seed (one split; too few units for a test)"}
        return paired_folds(out[a][key], out[b][key_b], folds.n_splits)

    if "fagi" in out:
        for o in ["gnn", "fagi_no_dp", "knn", "mlp_at", "mlp_dp"]:
            if o in out:
                for e in TEST_EPS:
                    tests[f"fagi_vs_{o}_fixed_{e}"] = cmp("fagi", o, f"fixed_{FPR_TARGET:g}_{e}")
                    tests[f"fagi_vs_{o}_pool_{e}"] = cmp("fagi", o, f"pool_{e}")
                    tests[f"fagi_vs_{o}_default_{e}"] = cmp("fagi", o, f"default_{e}")
                    for fpr in FPR_GRID:
                        tests[f"fagi_vs_{o}_fixed_{fpr:g}_{e}"] = cmp("fagi", o, f"fixed_{fpr:g}_{e}")
    for v in ["fagi_no_guard", "fagi_no_pgd"]:
        if v in out and "fagi" in out:
            for e in TEST_EPS:
                tests[f"fagi_vs_{v}_fixed_{e}"] = cmp("fagi", v, f"fixed_{FPR_TARGET:g}_{e}")
                tests[f"fagi_vs_{v}_pool_{e}"] = cmp("fagi", v, f"pool_{e}")
    for x, y in [("fagi_no_dp", "gnn"), ("fagi_no_dp", "knn"), ("knn", "gnn")]:
        if x in out and y in out:
            for e in TEST_EPS:
                tests[f"{x}_vs_{y}_fixed_{e}"] = cmp(x, y, f"fixed_{FPR_TARGET:g}_{e}")
                tests[f"{x}_vs_{y}_pool_{e}"] = cmp(x, y, f"pool_{e}")
                tests[f"{x}_vs_{y}_default_{e}"] = cmp(x, y, f"default_{e}")
                for fpr in FPR_GRID:
                    tests[f"{x}_vs_{y}_fixed_{fpr:g}_{e}"] = cmp(x, y, f"fixed_{fpr:g}_{e}")
    for m in out:
        for e in TEST_EPS:
            tests[f"{m}_full_vs_stored_fixed_{e}"] = cmp(m, m, f"fixed_{FPR_TARGET:g}_{e}", f"rescored_fixed_{FPR_TARGET:g}_{e}")
    return {"fpr_target": FPR_TARGET, "fpr_grid": FPR_GRID, "seeds": list(seeds), "eps": list(eps_list), "models": summ, "tests": tests,
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
    a = fixed_fpr(folds, eps_list=TEST_EPS)
    print("[1b] matched-FPR worst-case recall, all seeds, main models")
    a3 = fixed_fpr(folds, MAIN_MODELS, CONFIG["seeds"])
    print("[1c] matched-FPR worst-case recall, temporal split")
    at = fixed_fpr(CICIDSFolds(args.data_path, protocol="temporal"), TEMPORAL_MODELS, CONFIG["seeds"], eps_list=TEST_EPS)
    if args.skip_restarts and os.path.exists(args.output):
        with open(args.output) as fh:
            b = json.load(fh)["restart_curve"]
    else:
        print("[2] adaptive PGD restart curve at eps_adv = 0.5 (seed 0)")
        b = restart_curve(folds)
    save_json({"config_hash": folds.hash, "env": env_info(), "attack_suite": sa.SUITE, "fixed_fpr_seed0": a, "fixed_fpr": a3, "fixed_fpr_temporal": at,
               "restart_curve": b},
              args.output)


if __name__ == "__main__":
    main()
