#!/usr/bin/env bash
# Wait for verified isolated installation, then run ONE short probe.
# Never starts feature extraction or the main scientific training run.
set -u
experiment_root="$HOME/research/icl-dynamics/exp_001_hardware_check"
cd "$experiment_root" || exit 1
date -u +%FT%TZ > supervisor.started
deadline_epoch=$(( $(date +%s) + 14400 ))
while [ ! -f environment_ready ]; do
  if [ -f verified_install.exit_code ]; then
    echo 'Verified installation stopped before readiness; no model probe started.'
    cp verified_install.exit_code supervisor.exit_code
    exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline_epoch" ]; then
    printf '124\n' > supervisor.exit_code
    echo 'Installation wait exceeded four hours; no model probe started.'
    exit 124
  fi
  sleep 10
done
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
venv/bin/python -m pip freeze > environment.txt
venv/bin/python -m pip check || {
  printf '1\n' > supervisor.exit_code; exit 1;
}
bash run_benchmark_wsl.sh
code=$?
printf '%s\n' "$code" > supervisor.exit_code
date -u +%FT%TZ > supervisor.finished
exit "$code"
