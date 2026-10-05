"""Validate the saved baseline checkpoint before resuming author training."""
import hashlib
import json
from pathlib import Path
import sys
import argparse
import numpy as np
import h5py
import jax
import equinox as eqx

root=Path(__file__).resolve().parent
source=root.parent/'exp_001_hardware_check/source'
sys.path.insert(0,str(source))
import main_utils

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

config=root/'runs/main/config.json'
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=Path,default=root/'runs/main/checkpoints/00002950016.eqx')
parser.add_argument('--output',type=Path,default=root/'resume_audit.json')
args=parser.parse_args()
checkpoint=args.checkpoint
expected_iter=int(checkpoint.stem)
expected=json.loads((root/'features/feature_manifest.json').read_text())
for name,digest in expected.items():
    assert sha(root/'features'/name)==digest, name
with h5py.File(root/'runs/default_eval/eval_data.h5','r') as f:
    assert sorted(f.keys())==['flip_icl','icl','iwl_copy_avail','pure_iwl','train_eval']
    for key in f:
        assert f[key]['labels'].shape==(5000,3), (key,list(f[key]))
opts=main_utils.get_opts_from_json_file(config)
model=main_utils.get_model_from_opts(opts)
optimizer=main_utils.get_optimizer_from_opts(opts)
state=optimizer.init(eqx.filter(model,eqx.is_array))
key=jax.random.PRNGKey(0)
template=dict(iter=-1,model=model,opt_state=state,
              seeds=dict(eval_model_seed=key,train_data_seed=key,train_model_seed=key))
ckpt=eqx.tree_deserialise_leaves(checkpoint,template)
assert int(ckpt['iter'])==expected_iter
assert int(ckpt['opt_state'][0].count)==expected_iter//32
for array in jax.tree_util.tree_leaves(ckpt):
    if eqx.is_array(array):
        assert np.isfinite(np.asarray(array)).all()
report=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            checkpoint_bytes=checkpoint.stat().st_size,iter=int(ckpt['iter']),
            adam_updates=int(ckpt['opt_state'][0].count),
            configuration_sha256=sha(config),
            evaluation_data_sha256=sha(root/'runs/default_eval/eval_data.h5'),
            feature_sha256=expected,restores=['model','Adam state','all three PRNG keys'],
            audit='passed')
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
