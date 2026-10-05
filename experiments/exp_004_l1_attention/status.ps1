$ErrorActionPreference='Stop'
Get-ScheduledTask -TaskName 'icl-dynamics-exp004-attention' -ErrorAction SilentlyContinue|Select-Object TaskName,State|ConvertTo-Json -Compress
Get-ScheduledTaskInfo -TaskName 'icl-dynamics-exp004-attention' -ErrorAction SilentlyContinue|Select-Object LastTaskResult|ConvertTo-Json -Compress
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'cd ~/research/icl-dynamics/exp_004_l1_attention; cat exit_code 2>/dev/null || true; tail -8 logs/extract.log 2>/dev/null; tail -5 logs/plot.log 2>/dev/null; ps -eo pid,etime,args | grep "[p]ython -u extract_attention.py" || true; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader'
