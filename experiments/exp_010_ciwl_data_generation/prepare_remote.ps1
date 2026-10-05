$ErrorActionPreference='Stop'
$name='icl-dynamics-exp010-ciwl-generation'
$dir='D:\research\icl-dynamics\exp_010_ciwl_data_generation'
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {throw 'Generation task already exists'}
if (Test-Path "$dir\generation_task.started") {throw 'Generation has already started'}
New-Item -ItemType Directory -Path $dir -Force | Out-Null
[Environment]::MachineName
