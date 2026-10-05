#!/usr/bin/env bash
set -eu
root="$HOME/research/icl-dynamics/exp_003_sequence_classification"
probe="$HOME/research/icl-dynamics/exp_001_hardware_check"
baseline="$HOME/research/icl-dynamics/exp_002_main_reproduction"
cd "$root"
mkdir -p logs
date -u +%FT%TZ > replay.started
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
"$probe/venv/bin/python" -u replay_indices.py --source "$probe/source" --baseline "$baseline" --output "$root/pilot" --sequences 1000000 > logs/replay.log 2>&1
code=$?
printf '%s\n' "$code" > replay.exit_code
