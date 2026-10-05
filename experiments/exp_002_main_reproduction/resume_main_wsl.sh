#!/usr/bin/env bash
set -eu
root="$HOME/research/icl-dynamics/exp_002_main_reproduction"
probe="$HOME/research/icl-dynamics/exp_001_hardware_check"
cd "$root"
test -f data_order_approved.txt
test -f compute_route_approved.txt
mkdir -p logs
date -u +%FT%TZ > pipeline.started
echo "$$" > pipeline.linux_pid
finish() {
  code=$?
  if [ "$code" -eq 0 ] && [ "$(cat pipeline.stage 2>/dev/null)" != finished ]; then
    code=125
  fi
  printf '%s\n' "$code" > pipeline.exit_code
  date -u +%FT%TZ > pipeline.finished
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
python_executable="$probe/venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
printf 'audit_checkpoint\n' > pipeline.stage
"$python_executable" -u audit_resume.py > logs/resume_audit.log 2>&1
printf 'train_main_resume\n' > pipeline.stage
"$python_executable" -u run_main.py --source "$probe/source" --root "$root" --stage train --resume-checkpoint "$root/runs/main/checkpoints/00002950016.eqx" > logs/train_resume_00002950016.log 2>&1
printf 'plot_results\n' > pipeline.stage
"$python_executable" plot_results.py "$root/runs/main_resume_00002950016/log.h5" --prefix-log "$root/runs/main/log.h5" --output "$root/figures" > logs/plot.log 2>&1
printf 'finished\n' > pipeline.stage
