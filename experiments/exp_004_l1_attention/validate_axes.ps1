$ErrorActionPreference='Stop'
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'set -eu; cd ~/research/icl-dynamics/exp_004_l1_attention; cp /mnt/d/research/icl-dynamics/exp_004_l1_attention/validate_axes.py .; export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false; ~/research/icl-dynamics/exp_001_hardware_check/venv/bin/python -u validate_axes.py; cp outputs/axes_validation.json /mnt/d/research/icl-dynamics/exp_004_l1_attention/delivery/'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
