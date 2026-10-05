# Windows Task Scheduler keeps the WSL execution session alive after SSH exits.
$ErrorActionPreference='Stop'
$root='D:\research\icl-dynamics\exp_002_main_reproduction'
$bash=@'
cd ~/research/icl-dynamics/exp_002_main_reproduction || exit 1
bash prepare_features_wsl.sh
'@
(Get-Date).ToUniversalTime().ToString('o') | Set-Content "$root\features_task.started" -Encoding utf8
$p=[Diagnostics.Process]::new()
$p.StartInfo.FileName='C:\Program Files\WSL\wsl.exe'
$p.StartInfo.UseShellExecute=$false
$p.StartInfo.RedirectStandardOutput=$true
$p.StartInfo.RedirectStandardError=$true
$p.StartInfo.StandardOutputEncoding=[Text.Encoding]::UTF8
$p.StartInfo.StandardErrorEncoding=[Text.Encoding]::UTF8
foreach($a in @('-d','Ubuntu-22.04','--exec','bash','-lc',$bash)){
  $p.StartInfo.ArgumentList.Add($a)
}
[void]$p.Start()
$p.Id | Set-Content "$root\features_task.wsl_pid" -Encoding utf8
$out=$p.StandardOutput.ReadToEndAsync()
$err=$p.StandardError.ReadToEndAsync()
$p.WaitForExit()
@{exit_code=$p.ExitCode;stdout=$out.Result;stderr=$err.Result} | ConvertTo-Json -Depth 4 |
  Set-Content "$root\features_task.output.json" -Encoding utf8
$p.ExitCode | Set-Content "$root\features_task.exit_code" -Encoding utf8
(Get-Date).ToUniversalTime().ToString('o') | Set-Content "$root\features_task.finished" -Encoding utf8
exit $p.ExitCode
