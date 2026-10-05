$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Get-ScheduledTask -TaskName 'icl-dynamics-exp008-l2-ablation' -ErrorAction SilentlyContinue | Select-Object TaskName,State | ConvertTo-Json -Compress
Get-ScheduledTaskInfo -TaskName 'icl-dynamics-exp008-l2-ablation' -ErrorAction SilentlyContinue | Select-Object LastTaskResult | ConvertTo-Json -Compress
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'cd ~/research/icl-dynamics/exp_008_l2_ablation; cat exit_code 2>/dev/null || true; tail -5 logs/preflight.log 2>/dev/null || true; tail -5 logs/results.log 2>/dev/null || true; ls logs; pgrep -af "python.*visualize_runs" || true; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader'
