$ErrorActionPreference='Stop'
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'cd ~/research/icl-dynamics/exp_003_sequence_classification; cp /mnt/d/research/icl-dynamics/exp_003_sequence_classification/review_audit.py .; ~/research/icl-dynamics/exp_001_hardware_check/venv/bin/python review_audit.py --pilot pilot'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
