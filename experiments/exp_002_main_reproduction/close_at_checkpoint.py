"""Stop this one baseline only after its requested integer checkpoint exists."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime,timezone

root=Path(__file__).resolve().parent
target=20000000
checkpoint=root/f'runs/main_resume_00002950016/checkpoints/{target:011d}.eqx'
expected_bytes=json.loads((root/'resume_audit.json').read_text())['checkpoint_bytes']
matches=[]
for file in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args=file.read_bytes().split(b'\0')
        if b'run_main.py' in args and b'--resume-checkpoint' in args and str(root).encode() in args:
            matches.append(int(file.parent.name))
    except (FileNotFoundError,PermissionError,ProcessLookupError):
        continue
assert len(matches)==1,matches
pid=matches[0]
stat=Path(f'/proc/{pid}/stat')
def identity():
    try:
        return stat.read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None
started=identity()
report=dict(target_sequences=target,training_pid=pid,process_start_ticks=started,
            request='User requested an integer endpoint after observing the target behavior',
            state='waiting_for_complete_checkpoint',started_utc=datetime.now(timezone.utc).isoformat())
path=root/'close_at_20m.json'
def save():path.write_text(json.dumps(report,indent=2)+'\n')
save()
deadline=time.monotonic()+7200
while True:
    if identity()!=started:
        report['state']='training_process_disappeared_before_target';save()
        raise RuntimeError(report['state'])
    if checkpoint.exists() and checkpoint.stat().st_size==expected_bytes:
        time.sleep(.1)
        if checkpoint.stat().st_size==expected_bytes:
            break
    if time.monotonic()>deadline:
        report['state']='checkpoint_wait_timeout';save()
        raise TimeoutError(report['state'])
    time.sleep(.1)
# The checkpoint is written after evaluation/log close and before the next
# update. Preserve that exact saved boundary; any subsequent unsaved work is
# excluded from the frozen baseline. Never terminate a different/reused PID.
assert identity()==started
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
report.update(checkpoint=str(checkpoint),checkpoint_bytes=expected_bytes,
              checkpoint_sha256=digest,state='checkpoint_saved_sending_sigterm')
save()
os.kill(pid,signal.SIGTERM)
for _ in range(300):
    if identity()!=started:
        report.update(state='training_stopped_at_saved_boundary',
                      stopped_utc=datetime.now(timezone.utc).isoformat())
        save();print(json.dumps(report),flush=True);break
    time.sleep(.1)
else:
    report['state']='termination_not_confirmed';save()
    raise RuntimeError(report['state'])
