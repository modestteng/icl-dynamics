"""Independent checks for L1 identity, Q/K row-column meaning and aggregation."""
from pathlib import Path
import sys,json
root=Path.home()/'research/icl-dynamics'
work=root/'exp_004_l1_attention';base=root/'exp_002_main_reproduction'
sys.path.insert(0,str(root/'exp_001_hardware_check/source'))
sys.path.insert(1,str(work/'author_reference'))
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import main_utils,visualize_runs

opts=main_utils.get_opts_from_json_file(base/'runs/main/config.json')
m=main_utils.get_model_from_opts(opts);opt=main_utils.get_optimizer_from_opts(opts);key=jax.random.PRNGKey(0)
template=dict(iter=-1,model=m,opt_state=opt.init(eqx.filter(m,eqx.is_array)),
    seeds=dict(eval_model_seed=key,train_data_seed=key,train_model_seed=key))
with h5py.File(base/'runs/default_eval/eval_data.h5','r') as f:
    x=jnp.asarray(f['train_eval/examples'][:32]);y=jnp.asarray(f['train_eval/labels'][:32])
fwd=visualize_runs.make_forward_fn(opts)
report={'checks':[],'scope':'validate extraction axes and probabilities; no model intervention'}
for it in [4400000,20000000]:
    ck=eqx.tree_deserialise_leaves(base/f'runs/main_resume_00002950016/checkpoints/{it:011d}.eqx',template)
    model=ck['model'];batched=np.asarray(fwd(model,x,y,key)['activations'])
    full=model.call_with_all_aux(examples=x[0],labels=y[0],key=jax.random.split(key,32)[0],cache={},cache_mask={})
    l1=full['transformer_output']['block_outputs'][0]['attn_output']
    direct=np.asarray(l1['attn_scores'])[0]
    batch_error=float(np.abs(batched[0,0]-direct).max())
    assert batch_error<.002, batch_error
    qk=(l1['q']@jnp.transpose(l1['k'],(0,1,3,2)))*model.transformer.blocks[0].attn.scale
    mask=jnp.arange(5)[:,None]>=jnp.arange(5)[None,:]
    reconstructed=np.asarray(jax.nn.softmax(jnp.where(mask,qk,-1e20),axis=-1))[0]
    qk_error=float(np.abs(direct-reconstructed).max())
    assert qk_error<.002,qk_error
    with h5py.File(work/'outputs/attention.h5','r') as f:
        g=f[f'{it:011d}/train_eval']
        sample=np.asarray(g['sample0_attention'])[0]
        assert np.abs(sample-batched[0,0]).max()<1e-6
        mean=np.asarray(g['mean_attention'])[0];ci=np.asarray(g['correct_ind'])
        p=np.asarray(g['label_prev_per_sequence']);s=np.asarray(g['label_self_per_sequence'])
        p2=sum(.5*p[ci==c].mean(0,dtype=np.float64) for c in [0,1])
        s2=sum(.5*s[ci==c].mean(0,dtype=np.float64) for c in [0,1])
        np.testing.assert_allclose(p2,.5*(mean[:,1,0]+mean[:,3,2]),atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(s2,.5*(mean[:,1,1]+mean[:,3,3]),atol=1e-7,rtol=1e-6)
    report['checks'].append(dict(checkpoint=it,batch_vs_direct_max_error=batch_error,
        saved_sample_vs_batch_matches=True,qk_reconstruction_max_error=qk_error,
        layer_index=0,row_query_column_key=True,balanced_aggregation_verified=True,
        token_order=['x1','y1','x2','y2','xq'],sample0_labels=np.asarray(y[0]).tolist()))
report['status']='passed'
(work/'outputs/axes_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
