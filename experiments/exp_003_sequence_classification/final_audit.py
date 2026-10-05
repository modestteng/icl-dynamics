"""Validate finished artifacts and paired held-out effects using author evaluation.

Bootstrap intervals describe evaluation-sample uncertainty conditional on the
six trained branches; they are not intervals over training seeds or populations.
"""
from pathlib import Path
import hashlib
import json
import sys
from time import perf_counter

root=Path.home()/'research/icl-dynamics'
source=root/'exp_001_hardware_check/source'
baseline=root/'exp_002_main_reproduction'
pilot=root/'exp_003_sequence_classification/pilot'
sys.path.insert(0,str(source))
import numpy as np
import h5py
import jax
import equinox as eqx
import main_utils
import main
import opto

t0=perf_counter()
assert jax.default_backend()=='gpu'
assert jax.config.jax_default_matmul_precision is None

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

prov=json.loads((pilot/'score_provenance.json').read_text())
cfgpath=baseline/'runs/main/config.json'
evalpath=baseline/'runs/default_eval/eval_data.h5'
assert sha(cfgpath)==prov['config_sha256']
assert sha(evalpath)==prov['evaluation_sha256']
assert sha(pilot.parent/'score_and_validate.py')==prov['analysis_code_sha256']
assert sha(pilot.parent/'protocol.md')==prov['protocol_sha256']
for n,v in prov['source_sha256'].items():assert sha(source/n)==v
opts=main_utils.get_opts_from_json_file(cfgpath)
model=main_utils.get_model_from_opts(opts)
optimizer=main_utils.get_optimizer_from_opts(opts)
key=jax.random.PRNGKey(0)
template=dict(iter=-1,model=model,opt_state=optimizer.init(eqx.filter(model,eqx.is_array)),
    seeds=dict(eval_model_seed=key,train_data_seed=key,train_model_seed=key))
fwd=opto.make_fn_from_opts(opts)
validation=json.loads((pilot/'validation.json').read_text())
assert validation['status']=='completed' and len(validation['runs'])==6
assert len({(r['seed'],r['group']) for r in validation['runs']})==6
summary=json.loads((pilot/'classification_summary.json').read_text())
scores_audit={}
for it in [2000000,4400000,8000000]:
    with h5py.File(pilot/f'scores_{it:011d}.h5','r') as f:
        assert int(f.attrs['completed_sequences'])==1000000
        assert json.loads(f.attrs['provenance'])==prov
        arrays={n:np.asarray(f[n]) for n in f}
        assert all(v.shape==(1000000,) and np.isfinite(v).all() for v in arrays.values())
        expected=summary[str(it)]['counts']
        assert all(int((arrays['label']==k).sum())==expected[str(k)] for k in [0,1,2])
        assert np.array_equal(arrays['both_positive'],(arrays['icl_score']>0)&(arrays['ciwl_score']>0))
        scores_audit[str(it)]=dict(rows=1000000,finite=True,sha256=sha(pilot/f'scores_{it:011d}.h5'))
with h5py.File(evalpath,'r') as f:
    ev={n:{k:np.asarray(f[n][k][2500:5000]) for k in ['examples','labels']} for n in f}

def evaluate(m):
    out={}
    for n,data in ev.items():
        rows=[]
        for i in range(0,2500,32):
            r=main.eval_step(model=m,fwd_fn=fwd,x=data['examples'][i:i+32],y=data['labels'][i:i+32],key=key)
            rows.append({k:np.asarray(r[k]) for k in ['loss','acc','in_context_acc']})
        out[n]={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        assert all(x.shape==(2500,) and np.isfinite(x).all() for x in out[n].values())
    return out

anchor=eqx.tree_deserialise_leaves(baseline/'runs/main_resume_00002950016/checkpoints/00004400000.eqx',template)
anchor_metrics=evaluate(anchor['model'])
for n in anchor_metrics:
    for metric in anchor_metrics[n]:
        np.testing.assert_allclose(anchor_metrics[n][metric].mean(),validation['initial'][n][metric],rtol=1e-5,atol=1e-5)
branches={};checkpoints=[]
for r in validation['runs']:
    path=pilot/f"validation_{r['seed']}_{r['group']}.eqx"
    assert sha(path)==r['checkpoint_sha256']
    ck=eqx.tree_deserialise_leaves(path,template)
    assert int(ck['iter'])==4404096
    assert int(ck['opt_state'][0].count)==137628
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(ck) if eqx.is_array(x))
    ck_metrics=evaluate(ck['model'])
    for n in ck_metrics:
        for metric in ck_metrics[n]:
            np.testing.assert_allclose(ck_metrics[n][metric].mean(),r['after'][n][metric],rtol=1e-5,atol=1e-5)
    branches[(r['seed'],r['group'])]=ck_metrics
    checkpoints.append(dict(seed=r['seed'],group=r['group'],model_finite=True,adam_count=137628,
                            heldout_metrics_reproduced=True,sha256=sha(path)))
    print('final_checkpoint_audit',r['seed'],r['group'],flush=True)

rng=np.random.default_rng(5103)
bootstrap_ids=rng.integers(0,2500,size=(2000,2500),dtype=np.int32)
effects=[]
for seed in [103,104]:
    random=branches[(seed,'random')]
    for group,target in [('icl','icl'),('ciwl','iwl_copy_avail')]:
        selected=branches[(seed,group)]
        effect=dict(seed=seed,group=group,target=target,comparisons={})
        for n in selected:
            effect['comparisons'][n]={}
            for metric in ['loss','in_context_acc','acc']:
                # Positive values mean the selected branch outperformed random.
                r_values=random[n][metric].astype(np.float64)
                s_values=selected[n][metric].astype(np.float64)
                d=(r_values-s_values if metric=='loss' else s_values-r_values)
                ci=np.quantile(d[bootstrap_ids].mean(1),[.025,.975])
                effect['comparisons'][n][metric]=dict(advantage=float(d.mean()),
                    bootstrap_95_interval=ci.tolist())
        effects.append(effect)

gate_all_positive=all(e['comparisons'][e['target']]['loss']['advantage']>0 for e in effects)
assert gate_all_positive==validation['scale_gate_passed']
report=dict(status='passed_artifact_and_metric_audit',scores=scores_audit,checkpoints=checkpoints,
    scoring_provenance_verified=True,anchor_heldout_metrics_reproduced=True,
    paired_effects=effects,bootstrap_replicates=2000,bootstrap_seed=5103,
    interval_scope='conditional evaluation-sample uncertainty only; two selection seeds, one original training seed',
    elapsed_seconds=perf_counter()-t0,devices=[str(d) for d in jax.devices()])
(pilot/'final_audit.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(pilot/'heldout_metrics.npz',**{
    f'{seed}_{group}_{n}_{metric}':values
    for (seed,group),data in branches.items() for n,metrics in data.items() for metric,values in metrics.items()})
print(json.dumps(dict(status=report['status'],elapsed_seconds=report['elapsed_seconds'])),flush=True)
