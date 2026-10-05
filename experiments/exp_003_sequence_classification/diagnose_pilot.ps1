$ErrorActionPreference='Stop'
Get-ChildItem 'D:\research\icl-dynamics\exp_003_sequence_classification' | Select-Object Name,Length,LastWriteTime | ConvertTo-Json -Compress
Get-Content 'D:\research\icl-dynamics\exp_003_sequence_classification\pilot_task.output.json' -ErrorAction SilentlyContinue
Get-Content 'D:\research\icl-dynamics\exp_003_sequence_classification\pilot_task.wsl_pid' -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -like '*exp003*' -or $_.CommandLine -like '*exp_003*'} | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'whoami; echo "$HOME"; ls -la ~/research/icl-dynamics/exp_003_sequence_classification; ls -la /mnt/d/research/icl-dynamics/exp_003_sequence_classification; ps -eo pid,etime,pcpu,rss,args | head -25'
