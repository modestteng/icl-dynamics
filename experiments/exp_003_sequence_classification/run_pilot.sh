#!/usr/bin/env bash
set -euo pipefail
root="$HOME/research/icl-dynamics/exp_003_sequence_classification"
probe="$HOME/research/icl-dynamics/exp_001_hardware_check"
baseline="$HOME/research/icl-dynamics/exp_002_main_reproduction"
cd "$root"
mkdir -p logs pilot
date -u +%FT%TZ > pilot.started
finish() { code=$?; printf '%s\n' "$code" > pilot.exit_code; date -u +%FT%TZ > pilot.finished; }
trap finish EXIT
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
python="$probe/venv/bin/python"
if [ ! -f pilot/replay_audit.json ]; then
  printf 'replay\n' > pilot.stage
  resume=()
  if [ -f pilot/pilot_indices.h5 ]; then resume=(--resume); fi
  "$python" -u replay_indices.py --source "$probe/source" --baseline "$baseline" --output "$root/pilot" --sequences 1000000 "${resume[@]}" >> logs/replay.log 2>&1
fi
printf 'score_and_validate\n' > pilot.stage
"$python" -u score_and_validate.py --source "$probe/source" --baseline "$baseline" --output "$root/pilot" >> logs/score_and_validate.log 2>&1
printf 'completed\n' > pilot.stage
