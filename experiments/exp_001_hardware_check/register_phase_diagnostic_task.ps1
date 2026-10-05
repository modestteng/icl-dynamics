$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$name='icl-dynamics-exp001-diagnostic'
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
  throw 'Task already exists; inspect its state before changing it.'
}
$user=[Security.Principal.WindowsIdentity]::GetCurrent().Name
$action=New-ScheduledTaskAction -Execute (Get-Command pwsh).Source `
  -Argument '-NonInteractive -NoProfile -File "D:\research\icl-dynamics\exp_001_hardware_check\run_phase_diagnostic_task.ps1"'
$principal=New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings |
  Select-Object TaskName,State
Start-ScheduledTask -TaskName $name
Get-ScheduledTask -TaskName $name | Select-Object TaskName,State | ConvertTo-Json -Compress
