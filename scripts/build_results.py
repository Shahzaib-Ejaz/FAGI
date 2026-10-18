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

FILES = {
    "exp1": "exp1.json", "exp2": "exp2.json", "exp2_hard": "exp2_hard.json", "exp3": "exp3.json",
    "exp4": "exp4.json", "exp4_hard": "exp4_hard.json",
    "ablation": "ablation.json", "ablation_hard": "ablation_hard.json",
}


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        sys.exit(f"missing {p}; run scripts/run_all.sh first")
    with open(p) as f:
        return json.load(f)


def main():
    out = {"provenance": {"dataset_archive": DATASET_ARCHIVE,
                          "generated_by": "scripts/run_all.sh -> scripts/build_results.py"}}
    for key, fname in FILES.items():
        d = load(fname)
        d.pop("runs", None)
        out[key] = d
    out["provenance"]["cicids_config"] = out["exp1"]["config"]
    out["provenance"]["cicids_config_hash"] = out["exp1"]["config_hash"]
    out["provenance"]["synthetic_config"] = out["exp2"]["config"]
    out["provenance"]["env"] = out["exp1"]["env"]
    with open(os.path.join(RES, "paper_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(RES, "paper_results.json"))


if __name__ == "__main__":
    main()
