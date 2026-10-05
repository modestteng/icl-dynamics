#!/usr/bin/env bash
# Linux cloud launcher. On this Windows host, use Task Scheduler to hold WSL
# in the foreground: detached Linux jobs were terminated after SSH exited.
set -eu
experiment_root="${1:?experiment root required}"
python_executable="${2:?project interpreter required}"
stage="${3:?stage: encoder_probe, features, eval, train}"
source_root="${4:?unchanged author source directory required}"
adapter_root="$(cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p "$experiment_root/logs"
task_name="${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
case "$stage" in
  encoder_probe) command=("$python_executable" -u "$adapter_root/extract_features.py" --source "$source_root" --output "$experiment_root/features" --benchmark-only);;
  features) test -f "$experiment_root/data_order_approved.txt"; command=("$python_executable" -u "$adapter_root/extract_features.py" --source "$source_root" --output "$experiment_root/features" --data-order-approved);;
  eval) command=("$python_executable" -u "$adapter_root/run_main.py" --source "$source_root" --root "$experiment_root" --stage eval);;
  train) test -f "$experiment_root/compute_route_approved.txt"; command=("$python_executable" -u "$adapter_root/run_main.py" --source "$source_root" --root "$experiment_root" --stage train --paper-window);;
  *) echo 'Unknown stage' >&2; exit 2;;
esac
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
nohup setsid bash -c '
  log_prefix="$1"; shift
  date -u +%FT%TZ > "${log_prefix}.started"
  "$@"
  code=$?
  printf "%s\n" "$code" > "${log_prefix}.exit_code"
  date -u +%FT%TZ > "${log_prefix}.finished"
  exit "$code"
' _ "$experiment_root/logs/$task_name" "${command[@]}" \
  > "$experiment_root/logs/$task_name.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$experiment_root/logs/$task_name.pid"
printf 'task=%s pid=%s log=%s\n' "$task_name" "$pid" "$experiment_root/logs/$task_name.log"
