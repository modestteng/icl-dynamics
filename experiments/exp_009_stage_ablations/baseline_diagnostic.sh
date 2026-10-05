#!/usr/bin/env bash
# Check a recorded baseline discrepancy using the unchanged training evaluator.
set -euo pipefail
cd "$HOME/research/icl-dynamics/exp_009_stage_ablations"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$HOME/research/icl-dynamics/exp_001_hardware_check/source"
task_python="$HOME/research/icl-dynamics/exp_001_hardware_check/venv/bin/python"
trap 'printf "%s\n" "$?" > verification.exit_code' EXIT
test -e finished
"$task_python" - <<'PY' > logs/baseline_diagnostic.log 2>&1
from pathlib import Path
from functools import partial
from time import perf_counter
import hashlib,json
import numpy as np
import h5py,jax,jax.numpy as jnp,equinox as eqx
import main,main_utils,opto
t0=perf_counter();w=Path.cwd();root=w.parent
e=json.loads((w/'expected_baseline.json').read_text())
rows=json.loads((w/'outputs/summary.json').read_text());assert len(rows)==28
cfg=root/'exp_002_main_reproduction/runs/main/config.json'
ev=root/'exp_002_main_reproduction/runs/default_eval/eval_data.h5'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(cfg)==e['configuration_sha256'] and sha(ev)==e['evaluation_data_sha256']
for n,h in e['source_sha256'].items():assert sha(root/'exp_001_hardware_check/source'/n)==h
for stage,digest in e['checkpoint_sha256_by_stage'].items():
 assert sha(root/f'exp_002_main_reproduction/runs/main_resume_00002950016/checkpoints/{int(stage):011d}.eqx')==digest
opts=main_utils.get_opts_from_json_file(str(cfg));m0=main_utils.get_model_from_opts(opts,(512,))
assert opts.eval_bs==1000
key=jax.random.PRNGKey(0)
template=dict(iter=-1,model=m0,opt_state=main_utils.get_optimizer_from_opts(opts).init(eqx.filter(m0,eqx.is_array)),seeds=dict(eval_model_seed=key,train_data_seed=key,train_model_seed=key))
cp=root/'exp_002_main_reproduction/runs/main_resume_00002950016/checkpoints/00004000000.eqx'
ck=eqx.tree_deserialise_leaves(cp,template);model=ck['model']
with h5py.File(ev,'r') as f:
 x=jnp.asarray(f['iwl_copy_avail/examples'][:]);y=jnp.asarray(f['iwl_copy_avail/labels'][:])
fwd=opto.make_fn_from_opts(opts)
fn=main.make_batched_fn(partial(main.eval_step,fwd_fn=fwd),opts.eval_bs)
metrics={k:np.asarray(v) for k,v in fn(model,x,y,key=key).items()}
assert all(a.shape==(5000,) and np.isfinite(a).all() for a in metrics.values())
with h5py.File(w/'outputs/per_sequence_metrics.h5','r') as f:
 cli_acc=f['05_00004000000_baseline_iwl_copy_avail/in_context_acc'][:]
 cli_prob=f['05_00004000000_baseline_iwl_copy_avail/in_context_prob'][:]
a=metrics['in_context_acc'].astype(np.int8);b=cli_acc.astype(np.int8)
inds=np.flatnonzero(a!=b)
out=dict(stage=4000000,evaluator='iwl_copy_avail',author_training_eval_batch_size=1000,
 author_cli_batch_size=1024,native_accuracy=float(a.mean()),native_correct_count=int(a.sum()),
 cli_accuracy=float(b.mean()),cli_correct_count=int(b.sum()),
 baseline_logged_accuracy=e['baseline_logged_accuracy_by_stage']['4000000']['iwl_copy_avail'],
 author_native_evaluator='main.eval_step via main.make_batched_fn, unchanged',
 differing_judgments=[dict(row=int(i),native_correct=int(a[i]),cli_correct=int(b[i]),native_probability=float(metrics['in_context_prob'][i]),cli_probability=float(cli_prob[i])) for i in inds],
 native_array_file='baseline_native_4m_ciwl.npz',seconds=perf_counter()-t0,
 no_precision_or_parameter_change=True)
np.savez_compressed(w/'outputs/baseline_native_4m_ciwl.npz',**metrics)
(w/'outputs/baseline_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
print('All28 stored results retained. Scientific source/data/checkpoint identities verified after evaluation.',flush=True)
PY
