import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import env_info, mcnemar, mean_std, metrics, save_json
from synthetic_common import SYN_CONFIG, SyntheticData, load_sessions

TASKS = {
    "all_attacks": None,
    "lateral_movement": "lateral_movement",
    "sql_injection": "sql_injection",
}

TASK_MODELS = {
    "all_attacks": ["rf", "gnn_nopayload", "gnn", "fagi_no_payload", "fagi"],
    "lateral_movement": ["rf", "gnn_nopayload", "fagi"],
    "sql_injection": ["fagi_no_payload", "fagi"],
}


def task_indices(D, attack):
    if attack is None:
        return np.arange(len(D.y))
    return np.flatnonzero((D.attack_type == "none") | (D.attack_type == attack))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_data", default=None)
    ap.add_argument("--output", default="experiments/results/exp2.json")
    args = ap.parse_args()

    print("EXPERIMENT 2: API-native detection on the synthetic session dataset")
    D = SyntheticData(load_sessions(args.api_data))
    out = {"config": SYN_CONFIG, "env": env_info(), "n_sessions": int(len(D.y)), "tasks": {}}

    for task, attack in TASKS.items():
        idx = task_indices(D, attack)
        print(f"\n  task {task}: {len(idx)} sessions", flush=True)
        res = {m: {"acc": [], "adv": [], "fpr": [], "dr": [], "f1": [], "eps": []} for m in TASK_MODELS[task]}
        per_type = {m: {} for m in TASK_MODELS[task]}
        pooled = {m: [] for m in TASK_MODELS[task]}
        y_pool = []
        for f, (tr, te) in enumerate(D.subset_folds(idx)):
            y_pool.append(D.y[te])
            for sd in SYN_CONFIG["seeds"]:
                for m in TASK_MODELS[task]:
                    if m == "rf":
                        pc, adv, eps = D.rf(tr, te, seed=100 + 10 * f + sd), None, None
                    else:
                        model, info, _, Xte = D.train_variant(m, tr, te, seed_offset=1000 * f + 100 * sd)
                        r = D.evaluate(model, Xte, te, seed=7 + 10 * f + sd)
                        pc, adv, eps = r["pred_clean"], r["adv"], info.get("epsilon")
                    mm = metrics(D.y[te], pc)
                    res[m]["acc"].append(mm["accuracy"])
                    res[m]["fpr"].append(mm["fpr"])
                    res[m]["dr"].append(mm["detection_rate"])
                    res[m]["f1"].append(mm["f1"])
                    res[m]["adv"].append(adv)
                    res[m]["eps"].append(eps)
                    if sd == SYN_CONFIG["seeds"][0]:
                        pooled[m].append(pc)
                    for t in np.unique(D.attack_type[te]):
                        mask = D.attack_type[te] == t
                        want = 0 if t == "none" else 1
                        per_type[m].setdefault(t, []).append(float(np.mean(pc[mask] == want)))
            print(f"    fold {f + 1}: " + "  ".join(f"{m}={np.mean(res[m]['acc'][-len(SYN_CONFIG['seeds']):]):.3f}"
                                                  for m in TASK_MODELS[task]), flush=True)
        y_pool = np.concatenate(y_pool)
        tout = {"n_sessions": int(len(idx)), "models": {}}
        for m in TASK_MODELS[task]:
            r = res[m]
            tout["models"][m] = {
                "clean": mean_std(r["acc"]), "fpr": mean_std(r["fpr"]),
                "detection_rate": mean_std(r["dr"]), "f1": mean_std(r["f1"]),
                "adv": mean_std(r["adv"]) if any(a is not None for a in r["adv"]) else None,
                "epsilon": mean_std(r["eps"]) if any(e is not None for e in r["eps"]) else None,
                "per_type_detection": {t: float(np.mean(v)) for t, v in per_type[m].items()},
            }
            print(f"    {m:<16} {tout['models'][m]['clean']['mean'] * 100:6.2f} ± "
                  f"{tout['models'][m]['clean']['std'] * 100:4.2f}")
        if "fagi" in pooled and "fagi_no_payload" in pooled:
            tout["mcnemar_payload"] = mcnemar(y_pool, np.concatenate(pooled["fagi_no_payload"]),
                                              np.concatenate(pooled["fagi"]))
        out["tasks"][task] = tout

    save_json(out, args.output)


if __name__ == "__main__":
    main()
