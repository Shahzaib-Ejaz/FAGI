#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=experiments/results/logs
mkdir -p "$LOG"
PY="python3 -W ignore"

[ -f data/synthetic/sessions_final.json ] || $PY data/synthetic/generator.py
[ -f data/synthetic/sessions_hard.json ] || $PY data/synthetic/generator.py --difficulty hard --output data/synthetic/sessions_hard.json

$PY experiments/exp1_cicids_benchmark.py 2>&1 | tee "$LOG/exp1.log"
$PY experiments/ablation.py 2>&1 | tee "$LOG/ablation.log"
$PY experiments/exp3_adversarial_robustness.py 2>&1 | tee "$LOG/exp3.log"
$PY experiments/exp2_api_native.py 2>&1 | tee "$LOG/exp2.log"
$PY experiments/exp4_federated.py 2>&1 | tee "$LOG/exp4.log"

H=data/synthetic/sessions_hard.json
$PY experiments/exp2_api_native.py --api_data $H --output experiments/results/exp2_hard.json 2>&1 | tee "$LOG/exp2_hard.log"
$PY experiments/ablation.py --skip_cicids --api_data $H --output experiments/results/ablation_hard.json 2>&1 | tee "$LOG/ablation_hard.log"
$PY experiments/exp3_adversarial_robustness.py --skip_cicids --api_data $H --output experiments/results/exp3_hard.json 2>&1 | tee "$LOG/exp3_hard.log"
$PY experiments/exp4_federated.py --api_data $H --output experiments/results/exp4_hard.json 2>&1 | tee "$LOG/exp4_hard.log"

$PY scripts/build_results.py
$PY scripts/generate_figures.py
