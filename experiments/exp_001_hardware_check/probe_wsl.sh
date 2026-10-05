set -eu
id
uname -a
python3 --version
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
free -h
df -h /
python3 -c 'import importlib.util; print({n: bool(importlib.util.find_spec(n)) for n in ["jax", "torch", "h5py", "equinox", "pip", "venv"]})'
