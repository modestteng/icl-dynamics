$ProgressPreference='SilentlyContinue'
$ErrorActionPreference='Stop'
Get-ScheduledTask -TaskName 'icl-dynamics-exp006-l1-ablation-icl' | Select-Object TaskName,State | ConvertTo-Json -Compress
Get-ScheduledTaskInfo -TaskName 'icl-dynamics-exp006-l1-ablation-icl' | Select-Object LastTaskResult | ConvertTo-Json -Compress
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'set -eu; cd ~/research/icl-dynamics/exp_006_l1_ablation_icl; test "$(cat exit_code)" = 0; cat outputs/finished.json; sha256sum outputs/* > sha256.txt; tar -czf /mnt/d/research/icl-dynamics/exp_006_l1_ablation_icl/ablation_results.tar.gz outputs logs started finished exit_code sha256.txt protocol.md code_manifest.json; sha256sum /mnt/d/research/icl-dynamics/exp_006_l1_ablation_icl/ablation_results.tar.gz; cat started finished; ls -lh /mnt/d/research/icl-dynamics/exp_006_l1_ablation_icl/ablation_results.tar.gz'
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
