import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cicids_folds import CONFIG, CICIDSFolds

SOURCES = ["mlp", "gnn"]
MAIN = ["rf", "knn", "mlp_at", "cnn", "mlp_dp", "fagi", "fagi_no_dp"]
EXTRA = ["fagi_no_pgd", "fagi_no_guard"] + [f"fagi_sigma{s}" for s in CONFIG["sigma_sweep"]] + ["gnn_overfit"]
K100 = ["gnn", "fagi", "fagi_no_dp"]
TEMPORAL = ["rf", "knn", "mlp_at", "mlp_dp", "fagi", "fagi_no_dp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/cicids2017/")
    ap.add_argument("--protocol", choices=["cv", "temporal"], default="cv")
    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["seeds"])
    ap.add_argument("--extra", action="store_true", help="ablation variants, sigma sweep, MIA control, k100 (seed 0)")
    args = ap.parse_args()
    folds = CICIDSFolds(args.data_path, protocol=args.protocol)
    print(f"protocol {args.protocol}, config {folds.hash}, splits {folds.n_splits}", flush=True)
    for f in range(folds.n_splits):
        for s in args.seeds:
            for name in SOURCES:
                folds.result(name, f, s, k100=(args.extra and s == 0 and name in K100))
            names = MAIN if args.protocol == "cv" else TEMPORAL
            for name in names:
                folds.result(name, f, s, k100=(args.extra and s == 0 and name in K100))
            if args.extra and s == 0:
                for name in EXTRA:
                    folds.result(name, f, s)
                folds.result("gnn_fullpool", f, s)
            todo = SOURCES + names + (EXTRA if args.extra and s == 0 else [])
            for name in todo:
                if name != "gnn_overfit":
                    folds.transfer(name, f, s)


if __name__ == "__main__":
    main()
