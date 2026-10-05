#!/usr/bin/env bash
set -eu
root="$HOME/research/icl-dynamics/exp_002_main_reproduction"
probe="$HOME/research/icl-dynamics/exp_001_hardware_check"
cd "$root"
mkdir -p logs
python_executable="$probe/venv/bin/python"
"$python_executable" -m pip freeze > installed_constraints.txt
"$python_executable" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --constraint installed_constraints.txt --report encoder_install_report.json -r requirements_encoder.txt > logs/encoder_install.log 2>&1
"$python_executable" -m pip check
"$python_executable" -m pip freeze > environment_with_encoder.txt
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"$python_executable" -u extract_features.py --source "$probe/source" --output "$root/features" --data-order-approved > logs/features.log 2>&1
