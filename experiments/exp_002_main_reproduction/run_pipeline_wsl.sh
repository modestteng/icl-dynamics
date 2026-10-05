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
printf 'install_encoder\n' > pipeline.stage
"$python_executable" -m pip freeze > installed_constraints.txt
"$python_executable" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --constraint installed_constraints.txt --report encoder_install_report.json -r requirements_encoder.txt > logs/encoder_install.log 2>&1
"$python_executable" -m pip check > logs/pip_check.log 2>&1
"$python_executable" -m pip freeze > environment_with_encoder.txt
printf 'extract_features\n' > pipeline.stage
"$python_executable" -u extract_features.py --source "$probe/source" --output "$root/features" --data-order-approved > logs/features.log 2>&1
printf 'make_fixed_evaluation_sets\n' > pipeline.stage
"$python_executable" -u run_main.py --source "$probe/source" --root "$root" --stage eval > logs/eval.log 2>&1
printf 'train_main\n' > pipeline.stage
"$python_executable" -u run_main.py --source "$probe/source" --root "$root" --stage train > logs/train.log 2>&1
printf 'plot_results\n' > pipeline.stage
"$python_executable" plot_results.py "$root/runs/main/log.h5" --output "$root/figures" > logs/plot.log 2>&1
printf 'finished\n' > pipeline.stage
