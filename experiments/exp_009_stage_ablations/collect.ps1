$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Get-ScheduledTask -TaskName 'icl-dynamics-exp009-stage-ablations' | Select-Object TaskName,State | ConvertTo-Json -Compress
Get-ScheduledTaskInfo -TaskName 'icl-dynamics-exp009-stage-ablations' | Select-Object LastTaskResult | ConvertTo-Json -Compress
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'set -eu; cd ~/research/icl-dynamics/exp_009_stage_ablations; test "$(cat verification.exit_code)" = 0; cat outputs/metrics.tsv; sha256sum outputs/* logs/* author_reference/*.py run_cli.sh read_results.py protocol.md expected_baseline.json plan.json > sha256.txt; tar -czf /mnt/d/research/icl-dynamics/exp_009_stage_ablations/stage_results.tar.gz outputs logs runs/*/plots attempts started starts finished exit_code verification.exit_code sha256.txt protocol.md expected_baseline.json plan.json code_manifest.json; sha256sum /mnt/d/research/icl-dynamics/exp_009_stage_ablations/stage_results.tar.gz; cat started finished'
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
