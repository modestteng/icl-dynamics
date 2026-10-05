# Per-process request only; releases automatically when this experiment ends.
$ErrorActionPreference='Stop'
$root='D:\research\icl-dynamics\exp_002_main_reproduction'
Add-Type -Namespace ResearchCompute -Name Awake -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
$previous=[ResearchCompute.Awake]::SetThreadExecutionState([uint32]2147483649)
if ($previous -eq 0) { throw 'Failed to acquire system execution request' }
(Get-Date).ToUniversalTime().ToString('o') | Set-Content "$root\keep_awake.started" -Encoding utf8
try {
  while ((Get-ScheduledTask -TaskName 'icl-dynamics-exp002-main').State -eq 'Running') {
    Start-Sleep -Seconds 30
    [void][ResearchCompute.Awake]::SetThreadExecutionState([uint32]2147483649)
  }
} finally {
  [void][ResearchCompute.Awake]::SetThreadExecutionState([uint32]2147483648)
  (Get-Date).ToUniversalTime().ToString('o') | Set-Content "$root\keep_awake.finished" -Encoding utf8
}
