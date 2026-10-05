# Windows Task Scheduler keeps the WSL execution session alive after SSH exits.
$ErrorActionPreference='Stop'
$root='D:\research\icl-dynamics\exp_003_sequence_classification'
$bash=@'
cd ~/research/icl-dynamics/exp_003_sequence_classification || exit 1
bash run_pilot.sh
'@
(Get-Date).ToUniversalTime().ToString('o') | Set-Content "$root\pilot_task.started" -Encoding utf8
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
Add-Type -Namespace ResearchCompute -Name Awake -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
if ([ResearchCompute.Awake]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
  throw 'Failed to acquire the experiment system execution request'
}
try {
[void]$p.Start()
$p.Id | Set-Content "$root\pilot_task.wsl_pid" -Encoding utf8
$out=$p.StandardOutput.ReadToEndAsync()
$err=$p.StandardError.ReadToEndAsync()
$p.WaitForExit()
@{exit_code=$p.ExitCode;stdout=$out.Result;stderr=$err.Result} | ConvertTo-Json -Depth 4 |
  Set-Content "$root\pilot_task.output.json" -Encoding utf8
$p.ExitCode | Set-Content "$root\pilot_task.exit_code" -Encoding utf8
(Get-Date).ToUniversalTime().ToString('o') | Set-Content "$root\pilot_task.finished" -Encoding utf8
} finally {
  [void][ResearchCompute.Awake]::SetThreadExecutionState([uint32]2147483648)
}
exit $p.ExitCode
