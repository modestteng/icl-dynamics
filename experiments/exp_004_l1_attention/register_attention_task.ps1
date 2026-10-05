$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$name='icl-dynamics-exp004-attention'
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
  throw 'Task already exists; inspect its state before changing it.'
}
$user=[Security.Principal.WindowsIdentity]::GetCurrent().Name
$action=New-ScheduledTaskAction -Execute (Get-Command pwsh).Source `
  -Argument '-NonInteractive -NoProfile -WindowStyle Hidden -File "D:\research\icl-dynamics\exp_004_l1_attention\run_attention_task.ps1"'
$principal=New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $name -Action $action -Principal $principal -Settings $settings |
  Select-Object TaskName,State
Start-ScheduledTask -TaskName $name
Get-ScheduledTask -TaskName $name | Select-Object TaskName,State | ConvertTo-Json -Compress
