#!/usr/bin/env bash
# Prepared fallback; run only on a user-provided NVIDIA Linux server.
set -eu
adapter_root="$(cd -- "$(dirname -- "$0")" && pwd)"
experiment_root="${1:?independent experiment root required}"
python_bootstrap="${2:-python3}"
mkdir -p "$experiment_root"
"$python_bootstrap" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
"$python_bootstrap" -m venv "$experiment_root/venv"
python_executable="$experiment_root/venv/bin/python"
"$python_executable" -m pip install -r "$adapter_root/../exp_001_hardware_check/requirements_jax_cuda.txt" \
  -r "$adapter_root/requirements_encoder.txt" > "$experiment_root/install.log" 2>&1
"$python_executable" -m pip check
"$python_executable" -m pip freeze > "$experiment_root/environment.txt"
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda "$python_executable" -c \
  'import jax,torch; print(jax.devices()); assert jax.default_backend()=="gpu"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
