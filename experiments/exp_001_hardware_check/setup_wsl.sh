set -eu
experiment_root="$HOME/research/icl-dynamics/exp_001_hardware_check"
mkdir -p "$experiment_root"
cd "$experiment_root"
tar -xzf /mnt/d/research/icl-dynamics/exp_001_hardware_check/source.tar.gz
python3 -m venv --without-pip venv
# Ubuntu has system pip but no ensurepip. Bootstrap only this new venv.
bootstrap_modules="$(python3 -c 'import pathlib,pip; print(pathlib.Path(pip.__file__).parent.parent)')"
PYTHONPATH="$bootstrap_modules" venv/bin/python -m pip install --ignore-installed --no-deps 'pip==24.0' > bootstrap.log 2>&1
venv/bin/python -m pip install --disable-pip-version-check -r requirements_jax_cuda.txt > install.log 2>&1
venv/bin/python -m pip freeze > environment.txt
venv/bin/python -m pip check
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda venv/bin/python -c 'import jax; print(jax.__version__, jax.devices()); x=jax.numpy.ones(1); x.block_until_ready(); print(x.device)'
