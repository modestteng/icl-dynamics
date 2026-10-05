$ProgressPreference='SilentlyContinue'
$ErrorActionPreference='Stop'
$dir='D:\research\icl-dynamics\exp_010_ciwl_data_generation'
if ((Get-FileHash "$dir\input_bundle.tar.gz" -Algorithm SHA256).Hash.ToLower() -ne '8ee92d451da8ecb5ec8bbf5a24030b2f90a69620f101589aefcbea0f798062f7') {throw 'Input bundle hash mismatch'}
tar -xzf "$dir\input_bundle.tar.gz" -C $dir
if ($LASTEXITCODE -ne 0) {throw 'Windows extraction failed'}
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'set -eu; mkdir -p ~/research/icl-dynamics/exp_010_ciwl_data_generation; test ! -e ~/research/icl-dynamics/exp_010_ciwl_data_generation/started; tar -xzf /mnt/d/research/icl-dynamics/exp_010_ciwl_data_generation/input_bundle.tar.gz -C ~/research/icl-dynamics/exp_010_ciwl_data_generation; bash -n ~/research/icl-dynamics/exp_010_ciwl_data_generation/run_generation.sh'
if ($LASTEXITCODE -ne 0) {throw 'WSL material extraction failed'}
& pwsh -NoProfile -File "$dir\register_task.ps1"
if ($LASTEXITCODE -ne 0) {throw 'Task registration failed'}
