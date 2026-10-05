#!/usr/bin/env bash
# Call unchanged author data APIs; record their outputs without training a model.
set -euo pipefail
task_root="$HOME/research/icl-dynamics/exp_010_ciwl_data_generation"
research_root="$HOME/research/icl-dynamics"
cd "$task_root"
mkdir -p logs outputs
test ! -e outputs/ciwl_augmentation.h5
trap 'code=$?; printf "%s\n" "$code" > exit_code; date -u +%FT%TZ > finished' EXIT
date -u +%FT%TZ > started
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$research_root/exp_001_hardware_check/source"
task_python="$research_root/exp_001_hardware_check/venv/bin/python"
"$task_python" -m pip freeze > outputs/environment.txt
"$task_python" - <<'PY' > logs/generation.log 2>&1
from pathlib import Path
from functools import partial
from time import perf_counter
from datetime import datetime, timezone
import hashlib,json,platform
import numpy as np
import h5py,jax,jax.numpy as jnp,equinox
import main_utils,samplers

work=Path.cwd();root=work.parent;out=work/'outputs'
plan=json.loads((work/'plan.json').read_text())
expected=json.loads((work/'expected_baseline.json').read_text())
source=root/'exp_001_hardware_check/source'
cfgp=root/'exp_002_main_reproduction/runs/main/config.json'
ev=root/'exp_002_main_reproduction/runs/default_eval/eval_data.h5'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
started=perf_counter()
assert platform.node()=='LAPTOP-A9ON60MQ'
assert jax.default_backend()=='gpu'
assert jax.__version__=='0.4.26' and equinox.__version__=='0.11.4'
assert sha(cfgp)==expected['config_sha256']
assert sha(ev)==expected['evaluation_sha256']
for name,h in expected['source_sha256'].items():assert sha(source/name)==h,name
cfg=json.loads(cfgp.read_text());fp=Path(cfg['data_file'])
assert sha(fp)==expected['feature_sha256'][fp.name]
assert cfg['class_split']==[12800,0,184] and cfg['exemplar_split']==[20,0,0]
assert cfg['zipf_alpha']==0 and cfg['train_context_len']==2
assert cfg['fs_relabel']==0 and not cfg['match_query_and_distractors']
assert plan['seed'] not in [cfg['train_seed'],cfg['eval_seed']]
opts=main_utils.get_opts_from_json_file(str(cfgp))
data=main_utils.get_data_from_opts(opts)
assert data.shape==(12984,20,512) and data.dtype==jnp.float32
splits=main_utils.get_splits_from_opts(opts,data.shape)
n=plan['sequences'];key=jax.random.PRNGKey(plan['seed'])
class_sampler=partial(samplers.get_constant_burst_seq_idxs,
    classes=splits['class']['train'],class_distr=jnp.ones(12800)/12800,
    num_seqs=n,context_len=2,burstiness=0,distractor=True,
    no_support=True,unique_rest=True)
exemplar_sampler=partial(samplers.get_exemplar_inds,
    allowed_inds=splits['exemplar']['train'],match_query_and_distractors=False)
print('Identity verified; invoking author sampler',flush=True)

# Capture the outputs of the author APIs while the author make_data_sampler
# performs its unchanged label assignment and feature lookup.
def recorded_call(k,features):
    captured={}
    def record_classes(ck):
        r=class_sampler(ck);captured['class_idx']=r['class_idxs']
        captured['item_type']=r['idx_types'];return r
    def record_exemplars(ek,types):
        r=exemplar_sampler(ek,types);captured['exemplar_idx']=r;return r
    result=samplers.make_data_sampler(record_classes,record_exemplars,
        fs_relabel=None,noise_scale=0.,assign_query_label_random=True)(k,features)
    return result,captured
result,indices=jax.jit(recorded_call)(key,data)
result,indices=jax.tree_util.tree_map(np.asarray,(result,indices))
reference=jax.jit(samplers.make_data_sampler(class_sampler,exemplar_sampler,
    fs_relabel=None,noise_scale=0.,assign_query_label_random=True))(key,data)
for name in ['examples','labels']:
    np.testing.assert_array_equal(result[name],np.asarray(reference[name]))
x=result['examples'];y=result['labels'];c=indices['class_idx'];e=indices['exemplar_idx']
assert x.shape==(n,3,512) and y.shape==(n,3)
assert x.dtype==np.float32 and np.isfinite(x).all()
assert np.all((c>=0)&(c<12800)) and np.all((e>=0)&(e<20))
assert np.all(c[:,0]!=c[:,1]) and np.all(c[:,:2]!=c[:,-1,None])
assert np.all(y[:,-1]==c[:,-1])
match=y[:,:2]==y[:,-1,None]
assert np.all(match.sum(axis=1)==1)
correct_pos=np.argmax(match,axis=1).astype(np.uint8)
other_pos=1-correct_pos
np.testing.assert_array_equal(y[np.arange(n),other_pos],c[np.arange(n),other_pos])
reconstructed=np.asarray(data[jnp.asarray(c),jnp.asarray(e)])
np.testing.assert_array_equal(x,reconstructed)
assert np.all(indices['item_type'][:,:2]==int(samplers.ItemType.OTHER))
assert np.all(indices['item_type'][:,-1]==int(samplers.ItemType.QUERY))
print('Author reference and structural checks passed',flush=True)

def row_hash(xx,yy):
    h=hashlib.sha256(np.asarray(xx,dtype='<f4').tobytes())
    h.update(np.asarray(yy,dtype='<i4').tobytes());return h.digest()
hashes=[row_hash(xx,yy) for xx,yy in zip(x,y)]
assert len(set(hashes))==n,'Duplicate complete sequence inside generated data'
overlap={}
with h5py.File(ev,'r') as f:
    for name in f:
        old={row_hash(xx,yy) for xx,yy in zip(f[name]['examples'][:],f[name]['labels'][:])}
        overlap[name]=len(set(hashes)&old)
assert all(v==0 for v in overlap.values()),overlap

rows=[]
for i in range(n):
    rows.append({'sequence_id':i,'class_idx':[int(v) for v in c[i]],
        'exemplar_idx':[int(v) for v in e[i]],'labels':[int(v) for v in y[i]],
        'query_target':int(y[i,-1]),'correct_context_position':int(correct_pos[i]),
        'sequence_sha256':hashes[i].hex()})
with (out/'sequences.jsonl').open('w') as f:
    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
(out/'preview.json').write_text(json.dumps(rows[:8],ensure_ascii=False,indent=2)+'\n')
with h5py.File(out/'ciwl_augmentation.h5','w') as f:
    f.attrs['experiment_id']=plan['experiment_id'];f.attrs['seed']=plan['seed']
    f.attrs['role']='augmentation_training_candidates_not_evaluation'
    f.attrs['token_order']='[x1,y1,x2,y2,xq] -> labels[-1]'
    f.attrs['feature_sha256']=expected['feature_sha256'][fp.name]
    f.attrs['baseline_config_sha256']=expected['config_sha256']
    g=f.create_group(plan['output_group'])
    for name,array in [('examples',x),('labels',y.astype(np.int32)),
        ('class_idx',c.astype(np.uint16)),('exemplar_idx',e.astype(np.uint8)),
        ('item_type',indices['item_type'].astype(np.uint8)),
        ('correct_context_position',correct_pos),('sequence_id',np.arange(n,dtype=np.int64))]:
        g.create_dataset(name,data=array,compression='gzip',compression_opts=1,shuffle=True)
    f.flush()
with h5py.File(out/'ciwl_augmentation.h5','r') as f:
    for name,array in [('examples',x),('labels',y),('class_idx',c),('exemplar_idx',e)]:
        np.testing.assert_array_equal(f[plan['output_group']][name][:],array)
for name,h in expected['source_sha256'].items():assert sha(source/name)==h,name
assert sha(cfgp)==expected['config_sha256'] and sha(ev)==expected['evaluation_sha256']
assert sha(fp)==expected['feature_sha256'][fp.name]
elapsed=perf_counter()-started
audit={'status':'completed_data_generation_and_audit','sequences':n,
    'seed':plan['seed'],'feature_shape':list(x.shape),'label_shape':list(y.shape),
    'author_reference_exact_match':True,'source_feature_exact_match':True,
    'all_three_sample_classes_distinct':True,'query_class_absent_from_context':True,
    'query_fixed_label_exactly_once_in_context':True,'unmodified_other_context_label':True,
    'internal_duplicate_complete_sequences':0,'overlap_with_original_evaluators':overlap,
    'correct_context_position_counts':np.bincount(correct_pos,minlength=2).tolist(),
    'distinct_query_classes':int(len(np.unique(c[:,-1]))),
    'all_features_finite':True,'source_config_feature_eval_unchanged':True,
    'model_training_performed':False,'ciwl_improvement_measured':False,
    'execution_seconds':elapsed}
(out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
provenance={'created_utc':datetime.now(timezone.utc).isoformat(),
    'machine':platform.node(),'devices':[str(d) for d in jax.devices()],
    'jax':jax.__version__,'equinox':equinox.__version__,'plan':plan,
    'source_sha256':expected['source_sha256'],'baseline_config_sha256':sha(cfgp),
    'feature_path':str(fp),'feature_sha256':sha(fp),'original_eval_sha256':sha(ev),
    'generation_command':'bash run_generation.sh',
    'run_generation_sha256':sha(work/'run_generation.sh'),
    'protocol_sha256':sha(work/'protocol.md'),
    'sampler_call':'author CIWL evaluator style; one 5000-sequence batch; independent PRNGKey',
    'claim_scope':'paper-defined CIWL-only construction; no effect on learning tested'}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(audit,indent=2),flush=True)
PY
find outputs logs -type f -print0 | sort -z | xargs -0 sha256sum > sha256.txt
