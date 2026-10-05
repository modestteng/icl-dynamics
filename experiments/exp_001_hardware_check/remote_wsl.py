"""Run a recorded bash script in the existing Windows WSL2 environment.

Uses the external, pinned SSH configuration; never copies credentials.
This foreground runner is for setup and short probes, not long training jobs.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def run(script):
    encoded = base64.b64encode(script.encode()).decode()
    ps = r"""
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$p=[Diagnostics.Process]::new()
$p.StartInfo.FileName='C:\Program Files\WSL\wsl.exe'
$p.StartInfo.UseShellExecute=$false
$p.StartInfo.RedirectStandardOutput=$true
$p.StartInfo.RedirectStandardError=$true
$p.StartInfo.StandardOutputEncoding=[Text.Encoding]::UTF8
$p.StartInfo.StandardErrorEncoding=[Text.Encoding]::UTF8
$script=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PAYLOAD'))
foreach($a in @('-d','Ubuntu-22.04','--exec','bash','-lc',$script)){
  $p.StartInfo.ArgumentList.Add($a)
}
[void]$p.Start()
$out=$p.StandardOutput.ReadToEndAsync()
$err=$p.StandardError.ReadToEndAsync()
$p.WaitForExit()
$j=@{exit_code=$p.ExitCode;stdout=$out.Result;stderr=$err.Result} | ConvertTo-Json -Compress
[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($j))
""".replace("PAYLOAD", encoded)
    command = "pwsh -NonInteractive -NoProfile -EncodedCommand " + base64.b64encode(
        ps.encode("utf-16le")
    ).decode()
    result = subprocess.run(
        ["/usr/bin/ssh", "-o", "LogLevel=ERROR", "-F",
         "/Users/ai/.ssh/research-compute/ssh_config", "windows-compute", command],
        capture_output=True,
    )
    if result.returncode:
        report={"exit_code":result.returncode,
                "stdout":result.stdout.decode(errors="replace"),
                "stderr":result.stderr.decode(errors="replace")}
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        Path(__file__).with_name('transport_failure_'+stamp+'.json').write_text(
            json.dumps(report,indent=2)+'\n')
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return json.loads(base64.b64decode(result.stdout.strip()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.script.read_text())
    args.record.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["stdout"], end="")
    if report["exit_code"]:
        print(report["stderr"], file=sys.stderr)
    sys.exit(report["exit_code"])
