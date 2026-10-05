$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Get-ScheduledTask -TaskName 'icl-dynamics-exp010-ciwl-generation' -ErrorAction SilentlyContinue | Select-Object TaskName,State | ConvertTo-Json -Compress
Get-ScheduledTaskInfo -TaskName 'icl-dynamics-exp010-ciwl-generation' -ErrorAction SilentlyContinue | Select-Object LastTaskResult | ConvertTo-Json -Compress
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'cd ~/research/icl-dynamics/exp_010_ciwl_data_generation; cat exit_code 2>/dev/null || true; tail -4 logs/generation.log 2>/dev/null || true; tail -4 logs/generation.log 2>/dev/null || true; tail -4 logs/generation.log 2>/dev/null || true; ps -eo pid,etime,args | grep -E "[b]ash run_generation"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
