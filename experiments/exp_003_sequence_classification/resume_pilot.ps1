$ErrorActionPreference='Stop'
$task=Get-ScheduledTask -TaskName 'icl-dynamics-exp003-pilot'
if ($task.State -eq 'Running') { throw 'Task still running; inspect existing progress, do not duplicate it' }
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'cd ~/research/icl-dynamics/exp_003_sequence_classification; test "$(cat pilot.stage)" != completed || exit 42; if pgrep -f "[p]ython -u score_and_validate.py" >/dev/null; then exit 43; fi; test -f pilot/replay_audit.json; mkdir -p attempts; cp logs/score_and_validate.log "attempts/score_before_resume_$(date -u +%Y%m%dT%H%M%SZ).log"; rm -f pilot.exit_code pilot.finished'
if ($LASTEXITCODE -ne 0) { throw "Resume preflight refused, WSL exit=$LASTEXITCODE; inspect completed/live task instead" }
Start-ScheduledTask -TaskName 'icl-dynamics-exp003-pilot'
Get-ScheduledTask -TaskName 'icl-dynamics-exp003-pilot' | Select-Object TaskName,State | ConvertTo-Json -Compress
