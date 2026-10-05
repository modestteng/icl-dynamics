$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Get-ScheduledTask -TaskName 'icl-dynamics-exp008-l2-ablation' | Select-Object TaskName,State | ConvertTo-Json -Compress
Get-ScheduledTaskInfo -TaskName 'icl-dynamics-exp008-l2-ablation' | Select-Object LastTaskResult | ConvertTo-Json -Compress
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'set -eu; cd ~/research/icl-dynamics/exp_008_l2_ablation; test "$(cat exit_code)" = 0; cat outputs/metrics.tsv; sha256sum outputs/* logs/* author_reference/*.py run_cli.sh protocol.md expected_baseline.json > sha256.txt; tar -czf /mnt/d/research/icl-dynamics/exp_008_l2_ablation/ablation_results.tar.gz outputs logs runs/*/plots started finished exit_code sha256.txt protocol.md expected_baseline.json code_manifest.json; sha256sum /mnt/d/research/icl-dynamics/exp_008_l2_ablation/ablation_results.tar.gz; cat started finished'
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
