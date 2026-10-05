# Windows Task Scheduler keeps the WSL execution session alive after SSH exits.
$ErrorActionPreference='Stop'
$root='D:\research\icl-dynamics\exp_001_hardware_check'
$bash=@'
set -u
cd ~/research/icl-dynamics/exp_001_hardware_check || exit 1
date -u +%FT%TZ > persistent_task.started
echo "$$" > persistent_task.linux_pid
python3 -u install_verified_wheels.py > verified_install.log 2>&1
code=$?
printf '%s\n' "$code" > verified_install.exit_code
if [ "$code" -eq 0 ]; then
  bash run_benchmark_wsl.sh > probe_summary.log 2>&1
  code=$?
fi
printf '%s\n' "$code" > persistent_task.exit_code
date -u +%FT%TZ > persistent_task.finished
exit "$code"
'@
(Get-Date).ToUniversalTime().ToString('o') | Set-Content "$root\task.started" -Encoding utf8
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
$p.Id | Set-Content "$root\task.wsl_pid" -Encoding utf8
$out=$p.StandardOutput.ReadToEndAsync()
$err=$p.StandardError.ReadToEndAsync()
$p.WaitForExit()
@{exit_code=$p.ExitCode;stdout=$out.Result;stderr=$err.Result} | ConvertTo-Json -Depth 4 |
  Set-Content "$root\task.output.json" -Encoding utf8
$p.ExitCode | Set-Content "$root\task.exit_code" -Encoding utf8
(Get-Date).ToUniversalTime().ToString('o') | Set-Content "$root\task.finished" -Encoding utf8
exit $p.ExitCode
