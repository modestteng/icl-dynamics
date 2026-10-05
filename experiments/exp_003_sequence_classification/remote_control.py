"""Small transport helper; no credentials or scientific computation on the Mac."""
import argparse
import base64
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('script',type=Path)
a = p.parse_args()
ps = "$ProgressPreference='SilentlyContinue';\n"+a.script.read_text()
cmd = 'pwsh -NoProfile -NonInteractive -EncodedCommand '+base64.b64encode(ps.encode('utf-16le')).decode()
r = subprocess.run(['/usr/bin/ssh','-F','/Users/ai/.ssh/research-compute/ssh_config','windows-compute',cmd],capture_output=True)
print(r.stdout.decode('utf-8',errors='replace'))
if r.returncode:
    print(r.stderr.decode('utf-8',errors='replace'))
raise SystemExit(r.returncode)
