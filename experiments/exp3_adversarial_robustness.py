import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cicids_folds import CONFIG, CICIDSFolds
from cicids_stats import ms, paired
from common import env_info, predict, save_json
from synthetic_common import SYN_CONFIG, SyntheticData, inject_edges, load_sessions

SWEEP_MODELS = ["rf", "knn", "mlp", "mlp_at", "cnn", "gnn", "fagi_no_dp", "fagi"]
GRAPH = ["gnn", "fagi_no_dp", "fagi"]
EDGE_RATIOS = {"clean": 0.0, "edge_40pct": 0.4, "edge_80pct": 0.8}
STRUCT_MODELS = ["gnn", "fagi_no_guard", "fagi_no_dp", "fagi"]


def cicids_part(data_path):
    print("\n[Exp 3a] Evasion on CICIDS2017: malicious flows only, worst case over attacks")
    folds = CICIDSFolds(data_path)
    eps_list = [str(e) for e in CONFIG["eps_sweep"]]
    worst = {m: {e: [] for e in eps_list} for m in SWEEP_MODELS}
    by_attack = {m: {} for m in SWEEP_MODELS}
    strong, fixed01 = {m: [] for m in GRAPH}, {m: [] for m in GRAPH}
    for f in range(CONFIG["n_folds"]):
        for s in CONFIG["seeds"]:
            for m in SWEEP_MODELS:
                b = folds.result(m, f, s)
                for e in eps_list:
                    worst[m][e].append(float(b["worst_detected"][e].mean()))
                    for a, pred in b["attacks"][e].items():
                        by_attack[m].setdefault(a, {}).setdefault(e, []).append(float(np.mean(pred == 1)))
                if m in GRAPH:
                    strong[m].append(float(np.mean(b["attacks"]["strong_k100_fixed"] == 1)))
                    fixed01[m].append(float(np.mean(b["attacks"]["0.1"]["pgd_fixed_graph"] == 1)))
        print(f"  fold {f + 1}: " + "  ".join(f"{m}@0.5={np.mean(worst[m]['0.5'][-len(CONFIG['seeds']):]):.3f}"
                                              for m in ["gnn", "fagi", "knn", "mlp_at"]), flush=True)
    return {
        "eps_sweep_worst_recall": {m: {e: ms(v) for e, v in d.items()} for m, d in worst.items()},
        "by_attack_recall": {m: {a: {e: ms(v) for e, v in d.items()} for a, d in x.items()}
                             for m, x in by_attack.items()},
        "strong_k100_fixed_graph_eps0.1": {m: ms(v) for m, v in strong.items()},
        "k20_fixed_graph_eps0.1": {m: ms(v) for m, v in fixed01.items()},
        "tests": {f"fagi_vs_{o}_eps{e}": paired(worst["fagi"][e], worst[o][e])
                  for o in ["gnn", "knn", "mlp_at", "fagi_no_dp"] for e in ["0.1", "0.5"]},
    }


def structure_part(api_data):
    print("\n[Exp 3b] Random edge injection on synthetic API sessions")
    D = SyntheticData(load_sessions(api_data))
    idx = np.arange(len(D.y))
    res = {m: {c: [] for c in EDGE_RATIOS} for m in STRUCT_MODELS}
    drop = {m: [] for m in STRUCT_MODELS}
    for f, (tr, te) in enumerate(D.subset_folds(idx)):
        for s in SYN_CONFIG["seeds"]:
            for m in STRUCT_MODELS:
                model, _, _, Xte = D.train_variant(m, tr, te, seed_offset=1000 * f + 100 * s)
                for c, r in EDGE_RATIOS.items():
                    A = D.A[te] if r == 0 else inject_edges(D.A[te], r, seed=300 + 10 * f + s)
                    res[m][c].append(float(np.mean(predict(model, Xte, A) == D.y[te])))
                drop[m].append(res[m]["clean"][-1] - res[m]["edge_80pct"][-1])
        print(f"  fold {f + 1}: " + "  ".join(f"{m} drop80={drop[m][-1]:.4f}" for m in STRUCT_MODELS),
              flush=True)
    return {"accuracy": {m: {c: ms(v) for c, v in d.items()} for m, d in res.items()},
            "drop_at_80pct": {m: ms(v) for m, v in drop.items()},
            "tests": {"gnn_vs_fagi_no_guard_drop": paired(drop["gnn"], drop["fagi_no_guard"]),
                      "fagi_no_guard_vs_fagi_drop": paired(drop["fagi_no_guard"], drop["fagi"])}}


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
