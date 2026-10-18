import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cicids_folds import CONFIG, CICIDSFolds, config_hash
from common import adversarial_inputs, env_info, mcnemar, mean_std, predict, save_json
from synthetic_common import SYN_CONFIG, SyntheticData, inject_edges, load_sessions

EPS_SWEEP = [0.02, 0.05, 0.1, 0.2, 0.5]
STRONG_STEPS = 100
CICIDS_MODELS = ["gnn", "fagi_no_dp", "fagi"]
EDGE_RATIOS = {"clean": 0.0, "edge_40pct": 0.4, "edge_80pct": 0.8}


def cicids_part(data_path):
    print("\n[Exp 3a] Feature-space PGD on CICIDS2017 (attacker perturbs the query flow only)")
    folds = CICIDSFolds(data_path)
    sweep = {m: {e: [] for e in EPS_SWEEP} for m in CICIDS_MODELS}
    strong = {m: [] for m in CICIDS_MODELS}
    reretrieved = {m: [] for m in CICIDS_MODELS}
    pooled = {m: [] for m in CICIDS_MODELS}
    y_pool = []
    for f in range(CONFIG["n_folds"]):
        d = folds.fold(f)
        y = d["yte"]
        y_pool.append(y)
        for m in CICIDS_MODELS:
            model, blob = folds.graph_model(m, f)
            pooled[m].append(blob["pred_adv"])
            for e in EPS_SWEEP:
                xa = adversarial_inputs(model, d["Gte"], d["Ate"], y, node_mask=d["Mte"], eps=e,
                                        steps=CONFIG["pgd_eval_steps"], seed=100 + f)
                sweep[m][e].append(float(np.mean(predict(model, xa, d["Ate"]) == y)))
                if e == CONFIG["pgd_eps"]:
                    g2, a2, _ = d["builder"].build(xa[:, 0, :])
                    reretrieved[m].append(float(np.mean(predict(model, g2, a2) == y)))
            xs = adversarial_inputs(model, d["Gte"], d["Ate"], y, node_mask=d["Mte"], eps=CONFIG["pgd_eps"],
                                    steps=STRONG_STEPS, seed=200 + f)
            strong[m].append(float(np.mean(predict(model, xs, d["Ate"]) == y)))
        print(f"  fold {f + 1}: " + "  ".join(
            f"{m} eps0.1={sweep[m][0.1][-1]:.4f}/K100={strong[m][-1]:.4f}" for m in CICIDS_MODELS), flush=True)
    y_pool = np.concatenate(y_pool)
    return {
        "config_hash": config_hash(),
        "eps_sweep": {m: {str(e): mean_std(v) for e, v in sweep[m].items()} for m in CICIDS_MODELS},
        "strong_attack_k100_eps0.1": {m: mean_std(v) for m, v in strong.items()},
        "reretrieved_neighbours_eps0.1": {m: mean_std(v) for m, v in reretrieved.items()},
        "mcnemar_fagi_vs_gnn_adv": mcnemar(y_pool, np.concatenate(pooled["gnn"]), np.concatenate(pooled["fagi"])),
    }


def structure_part(api_data):
    print("\n[Exp 3b] Graph-structure attack on synthetic API sessions (random edge injection)")
    D = SyntheticData(load_sessions(api_data))
    idx = np.arange(len(D.y))
    models = ["gnn", "fagi_no_dp", "fagi"]
    res = {m: {c: [] for c in EDGE_RATIOS} for m in models}
    for f, (tr, te) in enumerate(D.subset_folds(idx)):
        for m in models:
            model, _, _, Xte = D.train_variant(m, tr, te, seed_offset=1000 * f)
            for c, r in EDGE_RATIOS.items():
                A = D.A[te] if r == 0 else inject_edges(D.A[te], r, seed=300 + f)
                res[m][c].append(float(np.mean(predict(model, Xte, A) == D.y[te])))
        print(f"  fold {f + 1}: " + "  ".join(
            f"{m} clean={res[m]['clean'][-1]:.3f}/+80%={res[m]['edge_80pct'][-1]:.3f}" for m in models), flush=True)
    return {m: {c: mean_std(v) for c, v in res[m].items()} for m in models}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--api_data", default=None)
    ap.add_argument("--output", default="experiments/results/exp3.json")
    ap.add_argument("--skip_cicids", action="store_true")
    args = ap.parse_args()
    print("EXPERIMENT 3: ADVERSARIAL ROBUSTNESS")
    out = {"env": env_info(), "cicids_config": CONFIG, "synthetic_config": SYN_CONFIG}
    if not args.skip_cicids:
        out["cicids2017"] = cicids_part(args.data_path)
    out["graph_structure_attack"] = structure_part(args.api_data)
    save_json(out, args.output)


if __name__ == "__main__":
    main()
