$ErrorActionPreference='Stop'
$task=Get-ScheduledTask -TaskName 'icl-dynamics-exp003-pilot'
if ($task.State -eq 'Running') { throw 'Task is still running; do not duplicate it' }
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'cd ~/research/icl-dynamics/exp_003_sequence_classification; mkdir -p attempts/attempt_001_default_precision; cp logs/score_and_validate.log pilot/score_provenance.json pilot.exit_code attempts/attempt_001_default_precision/; cp /mnt/d/research/icl-dynamics/exp_003_sequence_classification/score_and_validate.py /mnt/d/research/icl-dynamics/exp_003_sequence_classification/protocol.md .; rm -f pilot.exit_code pilot.finished; : > logs/score_and_validate.log'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Start-ScheduledTask -TaskName 'icl-dynamics-exp003-pilot'
Get-ScheduledTask -TaskName 'icl-dynamics-exp003-pilot' | Select-Object TaskName,State | ConvertTo-Json -Compress
