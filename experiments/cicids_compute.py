import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cicids_folds import CONFIG, CICIDSFolds

MAIN = ["rf", "knn", "mlp", "mlp_at", "cnn", "gnn", "fagi", "fagi_no_dp"]
EXTRA = ["fagi_no_pgd", "fagi_no_guard"] + [f"fagi_sigma{s}" for s in CONFIG["sigma_sweep"]] + ["gnn_overfit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["seeds"])
    ap.add_argument("--extra", action="store_true", help="ablation-only variants, sigma sweep, MIA control (seed 0)")
    args = ap.parse_args()
    folds = CICIDSFolds(args.data_path)
    print(f"config {folds.hash}")
    for f in range(CONFIG["n_folds"]):
        for s in args.seeds:
            for name in MAIN:
                folds.result(name, f, s)
        if args.extra:
            for name in EXTRA:
                folds.result(name, f, 0)


if __name__ == "__main__":
    main()
