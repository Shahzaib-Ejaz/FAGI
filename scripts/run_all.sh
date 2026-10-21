#!/usr/bin/env bash
# Regenerates every result, figure and paper_results.json from scratch.
# Expected runtime on a 2-core CPU: about 9 hours (CICIDS2017 grids ~5.5 h, synthetic ~3.5 h).
# Interrupted runs resume: CICIDS2017 models are cached per (model, fold, seed) under
# experiments/results/cache/, keyed by a hash of the config and the core source files.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=experiments/results/logs
mkdir -p "$LOG"
PY="python3 -W ignore"
H=data/synthetic/sessions_hard.json

[ -f data/synthetic/sessions_final.json ] || $PY data/synthetic/generator.py
[ -f "$H" ] || $PY data/synthetic/generator.py --difficulty hard --output "$H"

$PY experiments/cicids_compute.py --seeds 0 --extra 2>&1 | tee "$LOG/cicids_compute_s0.log"
$PY experiments/cicids_compute.py --seeds 1 2 2>&1 | tee "$LOG/cicids_compute_s12.log"
$PY experiments/cicids_compute.py --protocol temporal 2>&1 | tee "$LOG/cicids_compute_temporal.log"
$PY experiments/exp1_cicids_benchmark.py 2>&1 | tee "$LOG/exp1.log"
$PY experiments/mia_canary.py 2>&1 | tee "$LOG/mia_canary.log"
$PY experiments/exp5_temporal.py 2>&1 | tee "$LOG/exp5_temporal.log"
$PY experiments/robust_checks.py 2>&1 | tee "$LOG/robust_checks.log"
$PY experiments/ablation.py --skip_synthetic 2>&1 | tee "$LOG/ablation.log"
$PY experiments/exp3_adversarial_robustness.py --api_data $H 2>&1 | tee "$LOG/exp3.log"
$PY experiments/exp2_api_native.py 2>&1 | tee "$LOG/exp2.log"
$PY experiments/exp4_federated.py 2>&1 | tee "$LOG/exp4.log"

$PY experiments/exp2_api_native.py --api_data $H --output experiments/results/exp2_hard.json 2>&1 | tee "$LOG/exp2_hard.log"
$PY experiments/ablation.py --skip_cicids --api_data $H --output experiments/results/ablation_hard.json 2>&1 | tee "$LOG/ablation_hard.log"
$PY experiments/exp4_federated.py --api_data $H --output experiments/results/exp4_hard.json 2>&1 | tee "$LOG/exp4_hard.log"

$PY scripts/build_results.py
$PY scripts/generate_figures.py
$PY scripts/build_dashboard.py
