#!/usr/bin/env bash
set -euo pipefail
root="$HOME/research/icl-dynamics/exp_004_l1_attention"
cd "$root"
mkdir -p logs outputs
trap 'code=$?; printf "%s\n" "$code" > exit_code; date -u +%FT%TZ > finished' EXIT
date -u +%FT%TZ > started
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
python="$HOME/research/icl-dynamics/exp_001_hardware_check/venv/bin/python"
"$python" -u extract_attention.py > logs/extract.log 2>&1
"$python" -u plot_attention.py > logs/plot.log 2>&1
