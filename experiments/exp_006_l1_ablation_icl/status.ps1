$ErrorActionPreference='Stop'
Get-ScheduledTask -TaskName 'icl-dynamics-exp006-l1-ablation-icl' -ErrorAction SilentlyContinue|Select-Object TaskName,State|ConvertTo-Json -Compress
Get-ScheduledTaskInfo -TaskName 'icl-dynamics-exp006-l1-ablation-icl' -ErrorAction SilentlyContinue|Select-Object LastTaskResult|ConvertTo-Json -Compress
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'cd ~/research/icl-dynamics/exp_006_l1_ablation_icl; cat exit_code 2>/dev/null || true; tail -8 logs/eval.log 2>/dev/null; tail -5 logs/plot.log 2>/dev/null; ps -eo pid,etime,args | grep "[p]ython -u run_eval.py" || true; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader'
