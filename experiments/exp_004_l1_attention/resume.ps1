$ErrorActionPreference='Stop'
if ((Get-ScheduledTask -TaskName 'icl-dynamics-exp004-attention').State -eq 'Running') { throw 'Already running' }
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'set -eu; cd ~/research/icl-dynamics/exp_004_l1_attention; mkdir -p author_reference attempts; cp logs/extract.log attempts/attempt_001_missing_visualizer.log; cp /mnt/d/research/icl-dynamics/exp_004_l1_attention/{visualize_runs.py,plot_utils.py} author_reference/; cp /mnt/d/research/icl-dynamics/exp_004_l1_attention/extract_attention.py .; rm -f exit_code finished; ~/research/icl-dynamics/exp_001_hardware_check/venv/bin/python -c "import sys; sys.path.insert(0, str(__import__(\"pathlib\").Path.cwd()/\"author_reference\")); sys.path.insert(0, str(__import__(\"pathlib\").Path.home()/\"research/icl-dynamics/exp_001_hardware_check/source\")); import visualize_runs; print(visualize_runs.__file__)"'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-ScheduledTask -TaskName 'icl-dynamics-exp004-attention'
Get-ScheduledTask -TaskName 'icl-dynamics-exp004-attention'|Select-Object TaskName,State|ConvertTo-Json -Compress
