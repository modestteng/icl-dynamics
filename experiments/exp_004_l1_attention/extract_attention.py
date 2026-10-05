"""Read-only attention analysis using the author's forward and Fig14 averaging."""
from pathlib import Path
import hashlib,json,sys
from time import perf_counter
from functools import partial

root=Path.home()/'research/icl-dynamics'
work=root/'exp_004_l1_attention'
baseline=root/'exp_002_main_reproduction'
source=root/'exp_001_hardware_check/source'
sys.path.insert(0,str(source))
sys.path.insert(1,str(work/'author_reference'))
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import main_utils,opto,visualize_runs

t0=perf_counter()
assert jax.default_backend()=='gpu' and jax.config.jax_default_matmul_precision is None
outdir=work/'outputs';outdir.mkdir(exist_ok=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

expected=json.loads((work/'expected_baseline.json').read_text())
cfgpath=baseline/'runs/main/config.json';evalpath=baseline/'runs/default_eval/eval_data.h5'
assert sha(cfgpath)==expected['config_sha256']
assert sha(evalpath)==expected['evaluation_sha256']
source_locations={n:(source/n if (source/n).exists() else work/'author_reference'/n) for n in expected['source_sha256']}
for n,v in expected['source_sha256'].items():assert sha(source_locations[n])==v,n
opts=main_utils.get_opts_from_json_file(cfgpath)
model0=main_utils.get_model_from_opts(opts)
optimizer=main_utils.get_optimizer_from_opts(opts)
key=jax.random.PRNGKey(0)
template=dict(iter=-1,model=model0,opt_state=optimizer.init(eqx.filter(model0,eqx.is_array)),
    seeds=dict(eval_model_seed=key,train_data_seed=key,train_model_seed=key))
forward=visualize_runs.make_forward_fn(opts)
stages=[0,1000000,2000000,3000000,4400000,6000000,8000000,10000000,12000000,16000000,20000000]
sets=['train_eval','icl','iwl_copy_avail','flip_icl']
data={}
with h5py.File(evalpath,'r') as f:
    for name in sets:
        x=jnp.array(f[name]['examples']);y=jnp.array(f[name]['labels'])
        assert x.shape==(5000,3,512) and y.shape==(5000,3)
        matches=np.asarray(y[:,:2]==y[:,-1,None]);assert np.all(matches.sum(1)==1)
        ci=matches.argmax(1)
        data[name]=(x,y,ci)

def path(it):
    folder='main' if it<2950016 else 'main_resume_00002950016'
    return baseline/f'runs/{folder}/checkpoints/{it:011d}.eqx'

prov=dict(config_sha256=sha(cfgpath),eval_sha256=sha(evalpath),
    source_sha256=expected['source_sha256'],source_locations={n:str(p) for n,p in source_locations.items()},
    checkpoint_sha256={str(t):sha(path(t)) for t in stages},
    script_sha256=sha(Path(__file__)),protocol_sha256=sha(work/'protocol.md'),
    stages=stages,evaluators=sets,sequences_per_evaluator=5000,head_indexing='zero-based0..7',
    tensor_axes=['layer','head','attending_position','attended_position'],token_order=['x1','y1','x2','y2','xq'],
    averaging='equal weight correct_ind0/1, then average two label positions, matching Figure14',
    precision='float32, original default matmul precision',devices=[str(d) for d in jax.devices()],
    jax=jax.__version__,equinox=eqx.__version__)
(outdir/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
hp=outdir/'attention.h5'
summary={'stages':stages,'evaluators':sets,'results':{},'scope':'observational attention patterns; no causal intervention'}
with h5py.File(hp,'a') as f:
    if 'provenance' not in f.attrs:f.attrs['provenance']=json.dumps(prov,sort_keys=True)
    assert json.loads(f.attrs['provenance'])==prov
    for it in stages:
        ck=eqx.tree_deserialise_leaves(path(it),template)
        assert int(ck['iter'])==it and int(ck['opt_state'][0].count)==it//32
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(ck) if eqx.is_array(v))
        summary['results'][str(it)]={}
        for name in sets:
            groupname=f'{it:011d}/{name}'
            if groupname in f and f[groupname].attrs.get('completed',False):
                summary['results'][str(it)][name]=json.loads(f[groupname].attrs['summary'])
                continue
            if groupname in f:del f[groupname]
            x,y,ci=data[name];counts=np.bincount(ci,minlength=2)
            sums=np.zeros((2,2,8,5,5),np.float64)
            prev=np.empty((5000,8),np.float32);selfs=np.empty_like(prev)
            acc=np.empty(5000,np.float64);row_error=0.;sample0=None
            for start in range(0,5000,32):
                end=min(start+32,5000)
                r=forward(ck['model'],x[start:end],y[start:end],key)
                a=np.asarray(r['activations']);assert a.shape==(end-start,2,8,5,5)
                assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1+1e-6
                row_error=max(row_error,float(np.abs(a.sum(-1)-1).max()))
                assert row_error<2e-6
                assert float(a[...,np.triu_indices(5,1)[0],np.triu_indices(5,1)[1]].max())==0
                if sample0 is None:sample0=a[0].copy()
                for c in [0,1]:sums[c]+=a[ci[start:end]==c].sum(0,dtype=np.float64)
                prev[start:end]=.5*(a[:,0,:,1,0]+a[:,0,:,3,2])
                selfs[start:end]=.5*(a[:,0,:,1,1]+a[:,0,:,3,3])
                acc[start:end]=np.asarray(r['in_context_acc'])
                if it==0 and name=='train_eval' and start==0:
                    native={}
                    visualize_runs.update_all_token_layer_attention_over_time(native,it,r,ck['model'],{'correct_ind':jnp.array(ci[:end])})
                    test=sum(a[ci[:end]==c].mean(0,dtype=np.float64) for c in [0,1])/2
                    author=(np.asarray(native['activations_ind0'][0])+np.asarray(native['activations_ind1'][0]))/2
                    np.testing.assert_allclose(test,author,rtol=2e-6,atol=2e-7)
            cond=sums/counts[:,None,None,None,None]
            mean=cond.mean(0);unbalanced=sums.sum(0)/5000
            p=.5*(mean[0,:,1,0]+mean[0,:,3,2]);s=.5*(mean[0,:,1,1]+mean[0,:,3,3])
            se=lambda values:np.sqrt(sum(.25*values[ci==c].var(0,ddof=1)/counts[c] for c in [0,1]))
            result=dict(label_prev=p.tolist(),label_self=s.tolist(),prev_minus_self=(p-s).tolist(),
                label_prev_se=se(prev).tolist(),label_self_se=se(selfs).tolist(),
                in_context_acc=float(acc.mean()),conditioning_counts=counts.tolist(),
                row_sum_max_error=row_error)
            g=f.create_group(groupname)
            for n,v in [('mean_attention',mean),('mean_attention_unbalanced',unbalanced),('conditional_attention',cond),
                        ('sample0_attention',sample0),('sample0_labels',np.asarray(y[0])),
                        ('label_prev_per_sequence',prev),('label_self_per_sequence',selfs),('correct_ind',ci)]:
                g.create_dataset(n,data=v,compression='gzip')
            g.attrs['summary']=json.dumps(result);g.attrs['completed']=True;f.flush()
            summary['results'][str(it)][name]=result
            print('extracted',it,name,'acc',result['in_context_acc'],'elapsed',round(perf_counter()-t0,2),flush=True)
        (outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    comparisons={}
    for name in sets:
        g0=f[f'00004400000/{name}'];g1=f[f'00020000000/{name}']
        ci=np.asarray(g0['correct_ind']);counts=np.bincount(ci,minlength=2)
        p0=np.asarray(g0['label_prev_per_sequence']);p1=np.asarray(g1['label_prev_per_sequence'])
        s0=np.asarray(g0['label_self_per_sequence']);s1=np.asarray(g1['label_self_per_sequence'])
        d=(p0-s0)-(p1-s1)
        change=sum(.5*d[ci==c].mean(0,dtype=np.float64) for c in [0,1])
        se=np.sqrt(sum(.25*d[ci==c].var(0,ddof=1,dtype=np.float64)/counts[c] for c in [0,1]))
        before=summary['results']['4400000'][name];after=summary['results']['20000000'][name]
        comparisons[name]=[dict(head=h,prev_peak=before['label_prev'][h],self_peak=before['label_self'][h],
            prev_final=after['label_prev'][h],self_final=after['label_self'][h],
            dominance_switch=bool(before['prev_minus_self'][h]>0 and after['prev_minus_self'][h]<0
                and after['label_prev'][h]<before['label_prev'][h] and after['label_self'][h]>before['label_self'][h]),
            paired_dominance_change=float(change[h]),normal95_interval=[float(change[h]-1.96*se[h]),float(change[h]+1.96*se[h])]) for h in range(8)]
summary['peak_to_final']=comparisons
summary['elapsed_seconds']=perf_counter()-t0
(outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(outdir/'finished.json').write_text(json.dumps(dict(status='completed',elapsed_seconds=perf_counter()-t0,hdf5_sha256=sha(hp)),indent=2)+'\n')
print('completed',round(perf_counter()-t0,2),flush=True)
