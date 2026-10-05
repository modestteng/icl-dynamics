$ErrorActionPreference='Stop'
[Environment]::MachineName
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'ls -l ~/research/icl-dynamics/exp_002_main_reproduction/runs/main/checkpoints/00000000000.eqx; ls -l ~/research/icl-dynamics/exp_002_main_reproduction/runs/main_resume_00002950016/checkpoints/{00004400000,00020000000}.eqx; ls ~/research/icl-dynamics/exp_001_hardware_check/source/visualize_runs.py; mkdir -p ~/research/icl-dynamics/exp_004_l1_attention'
New-Item -ItemType Directory -Force 'D:\research\icl-dynamics\exp_004_l1_attention'|Out-Null
