$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Get-ScheduledTask -TaskName 'icl-dynamics-exp009-stage-ablations' -ErrorAction SilentlyContinue | Select-Object TaskName,State | ConvertTo-Json -Compress
Get-ScheduledTaskInfo -TaskName 'icl-dynamics-exp009-stage-ablations' -ErrorAction SilentlyContinue | Select-Object LastTaskResult | ConvertTo-Json -Compress
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'cd ~/research/icl-dynamics/exp_009_stage_ablations; cat exit_code 2>/dev/null || true; tail -4 logs/preflight.log 2>/dev/null || true; tail -4 logs/results.log 2>/dev/null || true; tail -4 logs/final_check.log 2>/dev/null || true; ps -eo pid,etime,args | grep -E "[p]ython.*visualize_runs|[b]ash run_cli"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
