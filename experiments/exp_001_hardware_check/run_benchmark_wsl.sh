set -eu
experiment_root="$HOME/research/icl-dynamics/exp_001_hardware_check"
cd "$experiment_root"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 1 > gpu_usage.csv &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
timeout 600 venv/bin/python -u benchmark.py --source source --out benchmark.json --steps 200 > benchmark.log 2>&1
cat benchmark.json
