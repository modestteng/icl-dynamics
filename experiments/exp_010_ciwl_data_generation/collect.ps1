$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Get-ScheduledTask -TaskName 'icl-dynamics-exp010-ciwl-generation' | Select-Object TaskName,State | ConvertTo-Json -Compress
Get-ScheduledTaskInfo -TaskName 'icl-dynamics-exp010-ciwl-generation' | Select-Object LastTaskResult | ConvertTo-Json -Compress
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'set -eu; cd ~/research/icl-dynamics/exp_010_ciwl_data_generation; test "$(cat exit_code)" = 0; cat outputs/audit.json; sha256sum outputs/* logs/* protocol.md plan.json expected_baseline.json code_manifest.json run_generation.sh > sha256.txt; tar -czf /mnt/d/research/icl-dynamics/exp_010_ciwl_data_generation/data_results.tar.gz outputs logs started finished exit_code sha256.txt protocol.md plan.json expected_baseline.json code_manifest.json run_generation.sh; sha256sum /mnt/d/research/icl-dynamics/exp_010_ciwl_data_generation/data_results.tar.gz; cat started finished'
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
