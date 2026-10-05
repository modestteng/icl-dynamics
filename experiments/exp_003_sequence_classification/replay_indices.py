"""Recover original training identifiers; no strategy-formation labels invented.

Reuses author sampling functions and the original batch size/PRNG schedule.
The pilot is the first 1M rows of the proposed first-10M candidate pool.
This preparation is independent of the pending classification criterion.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--sequences',type=int,default=1000000)
p.add_argument('--resume',action='store_true')
a=p.parse_args()
sys.path.insert(0,str(a.source.resolve()))
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import main,main_utils,samplers

assert jax.default_backend()=='gpu'
cfg_path=a.baseline/'runs/main/config.json'
cfg=json.loads(cfg_path.read_text())
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
manifest=json.loads((a.baseline/'features/feature_manifest.json').read_text())
assert file_sha(Path(cfg['data_file']))==manifest[Path(cfg['data_file']).name]
assert cfg['train_bs']==32 and a.sequences%32==0
assert cfg['class_split']==[12800,0,184] and cfg['exemplar_split']==[20,0,0]
assert cfg['train_context_len']==2 and cfg['fs_relabel']==0
assert cfg['noise_scale_train']==0 and not cfg['assign_query_label_random']
assert not cfg['match_query_and_distractors']
assert a.sequences==1000000,'Pilot stage only; expansion needs a recorded quality decision'
opts=main_utils.get_opts_from_json_file(cfg_path)
data=main_utils.get_data_from_opts(opts)
splits=main_utils.get_splits_from_opts(opts,data.shape)
ranks=jnp.arange(1,cfg['class_split'][0]+1,dtype=jnp.float32)
prob=1/ranks**cfg['zipf_alpha'];prob=prob/jnp.sum(prob)
substrates=[]
for i in range(len(cfg['mixing_coeffs'])):
    substrates.append(partial(samplers.get_constant_burst_seq_idxs,
        classes=splits['class']['train'],class_distr=prob,num_seqs=32,
        context_len=2,burstiness=cfg['pt_burstiness'][i],
        distractor=main.smart_index(cfg['pt_distract'],i,1),
        no_support=main.smart_index(cfg['pt_no_support'],i,0),
        unique_rest=main.smart_index(cfg['pt_unique_rest'],i,0)))
class_sampler=partial(samplers.get_mixed_seq_idxs,
    mix_probabilities=jnp.array(cfg['mixing_coeffs']),mix_substrate_fns=substrates)
exemplar_sampler=partial(samplers.get_exemplar_inds,
    allowed_inds=splits['exemplar']['train'],match_query_and_distractors=False)
reference=jax.jit(partial(samplers.make_data_sampler(class_sampler,exemplar_sampler,
    fs_relabel=None,noise_scale=0.,assign_query_label_random=False),data=data))

def index_batch(key):
    keys=jax.random.split(key,3)
    out=class_sampler(keys[0])
    ex=exemplar_sampler(keys[1],out['idx_types'])
    return out['class_idxs'],ex,out['idx_types']

def one(seed,_):
    seed,key=jax.random.split(seed)
    return seed,index_batch(key)

block64=jax.jit(lambda seed:jax.lax.scan(one,seed,None,length=64))
remainder=(a.sequences//32)%64
block_last=jax.jit(lambda seed:jax.lax.scan(one,seed,None,length=remainder))
a.output.mkdir(parents=True,exist_ok=True)
path=a.output/'pilot_indices.h5'
if path.exists() and not a.resume:raise FileExistsError(path)
seed=jax.random.split(jax.random.PRNGKey(cfg['train_seed']),2)[0]
started=perf_counter();reference_checks=0
with h5py.File(path,'a') as f:
    if 'class_idx' not in f:
        for name,dtype in [('class_idx','uint16'),('exemplar_idx','uint8'),('item_type','uint8')]:
            f.create_dataset(name,shape=(a.sequences,3),dtype=dtype,chunks=(2048,3))
        for name in ['exact_support_example','icl_viable','ciwl_viable']:
            f.create_dataset(name,shape=(a.sequences,),dtype='bool',chunks=(2048,))
        f.attrs['completed_sequences']=0
        f.attrs['train_data_seed']=np.asarray(seed)
        f.attrs['config_sha256']=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        f.attrs['sequence_ids']='row index = original global training sequence ID, zero based'
        f.attrs['label_semantics']='viability flags overlap; exact-match is a Section 7 data condition; not causal formation contribution'
    assert f.attrs['config_sha256']==hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    done=int(f.attrs['completed_sequences']);seed=jnp.asarray(f.attrs['train_data_seed'])
    assert done%32==0
    while done<a.sequences:
        before=seed
        use=min(64,(a.sequences-done)//32)
        seed,arrays=(block64(seed) if use==64 else block_last(seed))
        cls,ex,types=[np.asarray(v).reshape(-1,3) for v in arrays]
        matches=cls[:,:2]==cls[:,2,None]
        assert np.all(matches.sum(axis=1)==1)
        assert np.all(cls[:,0]!=cls[:,1])
        assert np.all((cls>=0)&(cls<12800)) and np.all((ex>=0)&(ex<20))
        exact=np.any(matches&(ex[:,:2]==ex[:,2,None]),axis=1)
        if done==0 or (done//2048)%100==0 or done+len(cls)==a.sequences:
            key=jax.random.split(before)[1]
            original=reference(key)
            assert np.array_equal(np.asarray(original['labels']),cls[:32])
            assert np.array_equal(np.asarray(original['examples']),np.asarray(data[cls[:32],ex[:32]]))
            reference_checks+=1
        end=done+len(cls)
        f['class_idx'][done:end]=cls;f['exemplar_idx'][done:end]=ex;f['item_type'][done:end]=types
        f['exact_support_example'][done:end]=exact
        f['icl_viable'][done:end]=True;f['ciwl_viable'][done:end]=True
        f.attrs['train_data_seed']=np.asarray(seed)
        f.attrs['completed_sequences']=end;f.flush();done=end
        if done%65536==0 or done==a.sequences:
            print('replayed',done,'elapsed_seconds',round(perf_counter()-started,2),flush=True)

# Check the replay's RNG boundary against a real training checkpoint at 1M.
model_opts=main_utils.get_opts_from_json_file(cfg_path)
model=main_utils.get_model_from_opts(model_opts)
optimizer=main_utils.get_optimizer_from_opts(model_opts)
state=optimizer.init(eqx.filter(model,eqx.is_array));key=jax.random.PRNGKey(0)
template=dict(iter=-1,model=model,opt_state=state,
              seeds=dict(eval_model_seed=key,train_data_seed=key,train_model_seed=key))
checkpoint=a.baseline/'runs/main/checkpoints/00001000000.eqx'
ckpt=eqx.tree_deserialise_leaves(checkpoint,template)
assert int(ckpt['iter'])==a.sequences
assert np.array_equal(np.asarray(seed),np.asarray(ckpt['seeds']['train_data_seed']))
with h5py.File(path,'r') as f:
    exact_count=int(np.asarray(f['exact_support_example']).sum())
report=dict(stage='index replay and structural audit only; strategy classification criterion pending',
    sequences=a.sequences,candidate_pool='proposed first 10M of original 20M; no expansion run yet',
    reference_batch_checks=reference_checks,rng_checkpoint_match=True,
    exact_match_count=exact_count,nonidentical_same_class_count=a.sequences-exact_count,
    both_strategies_viable_count=a.sequences,expected_exact_match_probability=1/20,
    elapsed_seconds=perf_counter()-started,devices=[str(d) for d in jax.devices()])
(a.output/'replay_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
