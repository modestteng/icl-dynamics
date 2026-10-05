set -eu
cd "$HOME/research/icl-dynamics/exp_001_hardware_check"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export JAX_LOG_COMPILES=1
set +e
timeout 180 venv/bin/python -u -c 'import faulthandler,runpy,sys; faulthandler.enable(); faulthandler.dump_traceback_later(60,repeat=True); sys.argv=["benchmark.py","--source","source","--out","diagnostic_benchmark.json","--steps","20"]; runpy.run_path("benchmark.py",run_name="__main__")' > phase_diagnostic.log 2>&1
code=$?
printf '%s\n' "$code" > phase_diagnostic.exit_code
exit "$code"
