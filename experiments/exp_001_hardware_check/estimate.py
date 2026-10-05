"""Extrapolate measured author-computation timings; never fabricate timings."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('benchmark',type=Path)
p.add_argument('--encoder',type=Path)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
b=json.loads(a.benchmark.read_text())
encoding=None
if a.encoder:
    encoding=json.loads(a.encoder.read_text())['estimated_full_encoding_seconds']
report={'machine':'Windows RTX 4070 Laptop / WSL2',
        'limitations':['Placeholder features only for timing, not scientific results.',
                       'Steady step measurements synchronize each update.',
                       'Encoding estimate must include the actual data pipeline.',
                       'HDF5 logging, preparation and thermal variation add overhead.'],
        'encoding_seconds':encoding, 'horizons':{}}
for sequences in [20000000,64000000]:
    updates=sequences//32
    evaluations=sequences//100000+1
    checkpoints=sequences//50000+1
    eval_seconds=evaluations*25*b['eval_mean_seconds']
    ckpt_seconds=checkpoints*b['checkpoint_seconds']
    fixed=b['setup_seconds']+b['warmup_seconds']+b['eval_compile_seconds']
    bounds=[]
    for seconds_per_step in [b['step_median_seconds'],b['step_p90_seconds']]:
        bounds.append(fixed+updates*seconds_per_step+eval_seconds+ckpt_seconds)
    total=None if encoding is None else [x+encoding for x in bounds]
    report['horizons'][str(sequences)]={
        'updates':updates,'evaluation_calls':evaluations,
        'checkpoint_writes':checkpoints,
        'compute_seconds_without_encoding':bounds,
        'total_seconds_estimate':total,
        'checkpoint_disk_bytes':checkpoints*b['checkpoint_bytes'],
        'route':'cloud_required_compute_alone_reaches_threshold' if min(bounds)>=10800 else
                'unknown_until_encoding_measured' if total is None else
                ('cloud_required_or_threshold_crossed' if max(total)>=10800 else
                 'windows_candidate_with_overhead_margin_still_required')}
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
