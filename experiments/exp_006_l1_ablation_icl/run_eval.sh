#!/usr/bin/env bash
set -euo pipefail
task_root="$HOME/research/icl-dynamics/exp_006_l1_ablation_icl"
cd "$task_root"
mkdir -p logs outputs
trap 'code=$?; printf "%s\n" "$code" > exit_code; date -u +%FT%TZ > finished' EXIT
date -u +%FT%TZ > started
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
task_python="$HOME/research/icl-dynamics/exp_001_hardware_check/venv/bin/python"
"$task_python" -m pip freeze > outputs/environment.txt
"$task_python" -u run_eval.py > logs/eval.log 2>&1
"$task_python" -u plot_results.py > logs/plot.log 2>&1
