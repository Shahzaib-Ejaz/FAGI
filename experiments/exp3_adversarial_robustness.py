import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from cicids_folds import CONFIG, CICIDSFolds
from cicids_stats import ms, paired
from common import env_info, predict, save_json
from synthetic_common import SYN_CONFIG, SyntheticData, inject_edges, load_sessions

SWEEP_MODELS = ["rf", "knn", "mlp", "mlp_at", "mlp_dp", "cnn", "gnn", "fagi_no_dp", "fagi"]
GRAPH = ["gnn", "fagi_no_dp", "fagi"]
DETERMINISTIC = {"knn"}
EDGE_RATIOS = {"clean": 0.0, "edge_40pct": 0.4, "edge_80pct": 0.8}
STRUCT_MODELS = ["gnn", "fagi_no_guard", "fagi_no_dp", "fagi"]


def fm(vals, nf, ns):
    return np.asarray(vals, float).reshape(nf, ns).mean(axis=1).tolist()


def cicids_part(data_path):
    print("\n[Exp 3a] Evasion on CICIDS2017: malicious flows only, worst case over all attacks incl. transfer")
    folds = CICIDSFolds(data_path)
    nf, seeds = folds.n_splits, CONFIG["seeds"]
    eps_list = [str(e) for e in CONFIG["eps_sweep"]]
    worst = {m: {e: [] for e in eps_list} for m in SWEEP_MODELS}
    by_attack = {m: {} for m in SWEEP_MODELS}
    k100, k20 = {m: {} for m in GRAPH}, {m: {} for m in GRAPH}
    for f in range(nf):
        for s in seeds:
            for m in SWEEP_MODELS:
                b = folds.result(m, f, s)
                w = folds.worst_detected(m, f, s)
                tr = folds.transfer(m, f, s)
                for e in eps_list:
                    worst[m][e].append(float(w[e].mean()))
                    for a, det in b["attacks"][e].items():
                        if not a.endswith("_check"):
                            by_attack[m].setdefault(a, {}).setdefault(e, []).append(float(det.mean()))
                    for src, by_e in tr.items():
                        by_attack[m].setdefault(f"transfer_from_{src}", {}).setdefault(e, []).append(float(by_e[e].mean()))
                if m in GRAPH and s == 0:
                    for e in [str(x) for x in CONFIG["k100_eps"]]:
                        k100[m].setdefault(e, []).append(float(b["attacks"][e]["pgd_adaptive_k100_check"].mean()))
                        k20[m].setdefault(e, []).append(float(b["attacks"][e]["pgd_adaptive"].mean()))
        print(f"  fold {f + 1}: " + "  ".join(f"{m}@0.5={np.mean(worst[m]['0.5'][-len(seeds):]):.3f}"
                                              for m in ["gnn", "fagi", "knn", "mlp_at", "mlp_dp"]), flush=True)

    def stat(m, v):
        return ms(fm(v, nf, len(seeds))) if m in DETERMINISTIC else ms(v)

    def cmp(a, b, e):
        if a in DETERMINISTIC or b in DETERMINISTIC:
            r = paired(fm(worst[a][e], nf, len(seeds)), fm(worst[b][e], nf, len(seeds)))
            r["unit"] = "fold (seeds averaged)"
        else:
            r = paired(worst[a][e], worst[b][e])
            r["unit"] = "fold-seed pair"
        return r

    return {
        "eps_sweep_worst_recall": {m: {e: stat(m, v) for e, v in d.items()} for m, d in worst.items()},
        "by_attack_recall": {m: {a: {e: stat(m, v) for e, v in d.items()} for a, d in x.items()}
                             for m, x in by_attack.items()},
        "adaptive_k100_vs_k20_seed0": {m: {e: {"k100": ms(k100[m][e]), "k20_restarts": ms(k20[m][e])}
                                           for e in k100[m]} for m in GRAPH},
        "tests": {f"fagi_vs_{o}_eps{e}": cmp("fagi", o, e)
                  for o in ["gnn", "knn", "mlp_at", "mlp_dp", "fagi_no_dp"] for e in ["0.1", "0.5"]},
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
        print(f"  fold {f + 1}: " + "  ".join(f"{m} drop80={drop[m][-1]:.4f}" for m in STRUCT_MODELS), flush=True)
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
    ap.add_argument("--skip_structure", action="store_true")
    ap.add_argument("--structure_from", default=None, help="reuse graph_structure_attack from an existing exp3.json")
    args = ap.parse_args()
    print("EXPERIMENT 3: ADVERSARIAL ROBUSTNESS")
    out = {"env": env_info(), "cicids_config": CONFIG, "synthetic_config": SYN_CONFIG,
           "structure_data": args.api_data or "data/synthetic/sessions_final.json"}
    if not args.skip_cicids:
        out["cicids2017"] = cicids_part(args.data_path)
    if args.structure_from:
        import json
        with open(args.structure_from) as f:
            old = json.load(f)
        out["graph_structure_attack"] = old["graph_structure_attack"]
        out["structure_data"] = old.get("structure_data", out["structure_data"])
    elif not args.skip_structure:
        out["graph_structure_attack"] = structure_part(args.api_data)
    save_json(out, args.output)


if __name__ == "__main__":
    main()
