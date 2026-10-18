import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")

DATASET_ARCHIVE = {
    "name": "CICIDS2017 MachineLearningCSV.zip (MachineLearningCVE/, 8 day files)",
    "source": "Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/ids-2017.html",
    "sha256": "c3f26274b36c837ccf28ffd2dbf4582941c30b3ee70a635c6e5b2f87c4727928",
    "md5": "4f83860afbf29cac8163854095bf6cf7",
}


def load(name, required=True):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        if required:
            sys.exit(f"missing {p}; run scripts/run_all.sh first")
        return None
    with open(p) as f:
        return json.load(f)


def m(x):
    return None if x is None else round(x["mean"], 4)


def s(x):
    return None if x is None else round(x["std"], 4)


def main():
    e1, e2, e3, e4, ab = (load("exp1.json"), load("exp2.json"), load("exp3.json"),
                          load("exp4.json"), load("ablation.json"))
    e2h, e3h, e4h, abh = (load("exp2_hard.json", False), load("exp3_hard.json", False),
                          load("exp4_hard.json", False), load("ablation_hard.json", False))

    rows = {r["key"]: r for r in e1["models"]}
    baselines = [{"name": r["name"], "key": r["key"], "clean": m(r["clean"]), "clean_std": s(r["clean"]),
                  "adv": m(r["adv"]), "adv_std": s(r["adv"]), "fpr": m(r["fpr"]),
                  "detection_rate": m(r["detection_rate"]), "f1": m(r["f1"]), "dp": r["dp"],
                  "epsilon": m(r["epsilon"])} for r in e1["models"]]
    abl_c = {r["key"]: r for r in ab["cicids2017"]}
    g, fa, fnd = rows["gnn"], rows["fagi"], abl_c["fagi_no_dp"]
    c3 = e3["cicids2017"]

    def ablation_rows(lst):
        return [{"name": r["name"], "key": r["key"], "clean": m(r["clean"]), "clean_std": s(r["clean"]),
                 "adv": m(r["adv"]), "adv_std": s(r["adv"]), "fpr": m(r["fpr"]), "dp": r["dp"],
                 "epsilon": m(r["epsilon"]),
                 **({"mia_auc": m(r["mia_auc"]), "mia_attack_accuracy": m(r["mia_attack_accuracy"])}
                    if "mia_auc" in r else {})} for r in lst]

    def api(e2x):
        if e2x is None:
            return None
        out = {}
        for task, t in e2x["tasks"].items():
            out[task] = {k: {"clean": m(v["clean"]), "clean_std": s(v["clean"]), "adv": m(v["adv"]),
                             "fpr": m(v["fpr"]), "epsilon": m(v["epsilon"]),
                             "per_type_detection": v["per_type_detection"]} for k, v in t["models"].items()}
            if "mcnemar_payload" in t:
                out[task]["mcnemar_payload"] = t["mcnemar_payload"]
        return out

    def gsa(e3x):
        if e3x is None:
            return None
        return {k: {c: m(v) for c, v in d.items()} for k, d in e3x["graph_structure_attack"].items()}

    def fed(e4x):
        if e4x is None:
            return None
        out = {}
        for k, v in e4x["summary"].items():
            out[k] = {"accuracy": m(v["accuracy"]), "accuracy_std": s(v["accuracy"])}
            for extra in ("fpr", "detection_rate", "epsilon", "mia_auc", "mia_attack_accuracy"):
                if extra in v:
                    out[k][extra] = m(v[extra])
            if "per_type_detection" in v:
                out[k]["per_type_detection"] = v["per_type_detection"]
        out["setup"] = {"train_orgs": e4x["train_orgs"], "test_org": e4x["test_org"], "rounds": e4x["rounds"],
                        "local_epochs": e4x["local_epochs"], "repeats": e4x["repeats"]}
        return out

    results = {
        "provenance": {
            "dataset_archive": DATASET_ARCHIVE,
            "cicids_config": e1["config"], "cicids_config_hash": e1["config_hash"],
            "synthetic_config": e2["config"], "env": e1["env"],
            "generated_by": "scripts/run_all.sh -> scripts/build_results.py",
        },
        "cicids2017": {
            "dataset": e1["dataset"], "n_features": e1["n_features"],
            "baselines": baselines,
            "per_class_detection": {k: rows[k]["per_class_detection"] for k in rows},
            "per_class_detection_adv": {k: rows[k]["per_class_detection_adv"] for k in rows
                                        if "per_class_detection_adv" in rows[k]},
            "adversarial_robustness": {
                "gnn_clean": m(g["clean"]), "gnn_adv": m(g["adv"]),
                "fagi_clean": m(fa["clean"]), "fagi_adv": m(fa["adv"]),
                "fagi_no_dp_clean": m(fnd["clean"]), "fagi_no_dp_adv": m(fnd["adv"]),
                "gap_pp": round(100 * (m(fa["adv"]) - m(g["adv"])), 2),
                "gnn_drop_pp": round(100 * (m(g["clean"]) - m(g["adv"])), 2),
                "fagi_drop_pp": round(100 * (m(fa["clean"]) - m(fa["adv"])), 2),
                "mcnemar_adv": e1["mcnemar_fagi_vs_gnn_adv"], "mcnemar_clean": e1["mcnemar_fagi_vs_gnn_clean"],
            },
            "eps_sweep": {k: {e: m(v) for e, v in d.items()} for k, d in c3["eps_sweep"].items()},
            "strong_attack_k100": {k: m(v) for k, v in c3["strong_attack_k100_eps0.1"].items()},
            "reretrieved_neighbours": {k: m(v) for k, v in c3["reretrieved_neighbours_eps0.1"].items()},
        },
        "api_native": {"original": api(e2), "hard": api(e2h)},
        "graph_structure_attack": {"original": gsa(e3), "hard": gsa(e3h)},
        "ablation": {"cicids2017": ablation_rows(ab["cicids2017"]),
                     "synthetic": ablation_rows(ab["synthetic"]),
                     "synthetic_hard": ablation_rows(abh["synthetic"]) if abh else None},
        "federated": {"original": fed(e4), "hard": fed(e4h)},
        "privacy": {
            "mechanism": "DP-SGD (Abadi et al. 2016): per-example clipping, Gaussian noise, Poisson sampling",
            "accountant": "Renyi DP (Mironov 2017; Mironov, Talwar, Zhang 2019) via Opacus compute_rdp",
            "sigma": e1["config"]["dp_sigma"], "clip_norm": e1["config"]["dp_clip"], "delta": e1["config"]["dp_delta"],
            "epsilon_cicids2017": m(fa["epsilon"]),
            "epsilon_synthetic_central": m(e2["tasks"]["all_attacks"]["models"]["fagi"]["epsilon"]),
            "epsilon_federated_per_client": (fed(e4) or {}).get("fedavg_dp", {}).get("epsilon"),
        },
    }
    out = os.path.join(RES, "paper_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
