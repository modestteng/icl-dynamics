"""Checkpoint-local first-order scores, followed by actual author Adam validation.

Not a paper-provided classifier; never label scores as exact Adam contributions.
All scientific model, data and training functions come from the frozen source.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--checkpoints', nargs='+', type=int, default=[2000000, 4400000, 8000000])
p.add_argument('--limit', type=int, default=1000000)
p.add_argument('--audit-only', action='store_true')
p.add_argument('--score-matmul-precision', choices=['default','highest'], default='highest')
a = p.parse_args()
sys.path.insert(0, str(a.source.resolve()))
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import main
import main_utils
import opto

assert jax.default_backend() == 'gpu', jax.devices()
original_matmul_precision = jax.config.jax_default_matmul_precision
jax.config.update('jax_default_matmul_precision', a.score_matmul_precision)
assert a.limit == 1000000 or a.audit_only
a.output.mkdir(parents=True, exist_ok=True)
t0 = perf_counter()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    tmp.replace(path)

cfgpath = a.baseline/'runs/main/config.json'
cfg = json.loads(cfgpath.read_text())
opts = main_utils.get_opts_from_json_file(cfgpath)
assert cfg['train_bs'] == 32 and cfg['weight_decay'] == 0
manifest = json.loads((a.baseline/'features/feature_manifest.json').read_text())
assert sha(cfg['data_file']) == manifest[Path(cfg['data_file']).name]
data = main_utils.get_data_from_opts(opts)
template_model = main_utils.get_model_from_opts(opts)
optimizer = main_utils.get_optimizer_from_opts(opts)
key = jax.random.PRNGKey(0)
template = dict(iter=-1, model=template_model,
                opt_state=optimizer.init(eqx.filter(template_model, eqx.is_array)),
                seeds=dict(eval_model_seed=key, train_data_seed=key, train_model_seed=key))
fwd = opto.make_fn_from_opts(opts)
evalpath = a.baseline/'runs/default_eval/eval_data.h5'
with h5py.File(evalpath, 'r') as f:
    evaluators = {n: {k: jnp.array(f[n][k]) for k in ['examples', 'labels']} for n in f}
    assert all(v['labels'].shape == (5000, 3) for v in evaluators.values())
    assert set(evaluators) == {'train_eval', 'icl', 'iwl_copy_avail', 'flip_icl', 'pure_iwl'}
assert np.all(np.isin(np.asarray(evaluators['icl']['labels']), [0, 1]))
with h5py.File(a.output/'pilot_indices.h5', 'r') as f:
    assert int(f.attrs['completed_sequences']) == 1000000
    classes = np.asarray(f['class_idx'])
    exemplars = np.asarray(f['exemplar_idx'])
    exact = np.asarray(f['exact_support_example'])
    assert f.attrs['config_sha256'] == sha(cfgpath)
replay_audit = json.loads((a.output/'replay_audit.json').read_text())
assert replay_audit['rng_checkpoint_match']
matches = classes[:, :2] == classes[:, 2, None]
assert np.all(matches.sum(1) == 1)
assert np.array_equal(exact, np.any(matches & (exemplars[:, :2] == exemplars[:, 2, None]), axis=1))

def ckpt_path(it):
    folder = 'main' if it < 2950016 else 'main_resume_00002950016'
    return a.baseline/f'runs/{folder}/checkpoints/{it:011d}.eqx'

def load(it):
    ck = eqx.tree_deserialise_leaves(ckpt_path(it), template)
    assert int(ck['iter']) == it
    assert int(ck['opt_state'][0].count) == it//32
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(ck) if eqx.is_array(x))
    return ck

def samples(ids):
    c = jnp.asarray(classes[ids], dtype=jnp.int32)
    e = jnp.asarray(exemplars[ids], dtype=jnp.int32)
    return data[c, e], c

def losses(model, x, y):
    keys = jax.random.split(key, len(x))
    out = jax.vmap(partial(fwd, model=model))(x=x, y=y, key=keys)['out']
    return main.ce(out[:, -1, :], y[:, -1])

grad_fn = eqx.filter_jit(eqx.filter_grad(lambda m, x, y: losses(m, x, y).mean()))
loss_fn = eqx.filter_jit(losses)

def dot(g, h):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(h)))

def direction(model, ev):
    total = None
    for i in range(0, 2500, 32):
        x = ev['examples'][i:min(i+32, 2500)]
        y = ev['labels'][i:min(i+32, 2500)]
        g = grad_fn(model, x, y)
        w = len(x)/2500
        g = jax.tree_util.tree_map(lambda z: z*w, g)
        total = g if total is None else jax.tree_util.tree_map(lambda z, q: z+q, total, g)
    norm = float(jnp.sqrt(dot(total, total)))
    assert np.isfinite(norm) and norm > 0
    return jax.tree_util.tree_map(lambda z: z/norm, total), norm

@eqx.filter_jit
def score(model, d1, d2, x, y):
    fun = lambda m: losses(m, x, y)
    l, s1 = eqx.filter_jvp(fun, (model,), (d1,))
    _, s2 = eqx.filter_jvp(fun, (model,), (d2,))
    return l, s1, s2

def classify(s1, s2):
    den = np.abs(s1)+np.abs(s2)
    diff = (s1-s2)/(den+1e-12)
    label = np.zeros(len(s1), np.int8)
    label[(diff > 0.1) & (s1 > 0) & (den > 1e-10)] = 1
    label[(diff < -0.1) & (s2 > 0) & (den > 1e-10)] = 2
    return diff, label

provenance = dict(config_sha256=sha(cfgpath), evaluation_sha256=sha(evalpath),
    source_sha256={n: sha(a.source/n) for n in ['main.py','main_utils.py','models.py','samplers.py','opto.py']},
    feature_sha256=manifest, checkpoint_sha256={str(it): sha(ckpt_path(it)) for it in a.checkpoints},
    devices=[str(d) for d in jax.devices()], jax=jax.__version__, equinox=eqx.__version__,
    analysis_code_sha256=sha(Path(__file__)),
    scoring_matmul_precision=a.score_matmul_precision, validation_matmul_precision=original_matmul_precision,
    scoring='normalized evaluation-gradient inner products; local SGD proxy, not exact Adam',
    evaluation_split='first2500 score, last2500 validation', protocol_sha256=sha(a.output.parent/'protocol.md'))
dump(a.output/'score_provenance.json', provenance)
summaries = {}
for it in a.checkpoints:
    ck = load(it)
    model = ck['model']
    d1, norm1 = direction(model, evaluators['icl'])
    d2, norm2 = direction(model, evaluators['iwl_copy_avail'])
    x, y = samples(np.arange(32))
    l, b1, b2 = [np.asarray(v) for v in score(model, d1, d2, x, y)]
    discrepancies = []
    for i in [0, 7, 31]:
        g = grad_fn(model, x[i:i+1], y[i:i+1])
        direct = np.array([float(dot(g, d1)), float(dot(g, d2))])
        batch = np.array([b1[i], b2[i]])
        np.testing.assert_allclose(batch, direct, rtol=2e-3, atol=3e-4)
        single = score(model, d1, d2, x[i:i+1], y[i:i+1])
        np.testing.assert_allclose(batch, [float(single[1][0]),float(single[2][0])], rtol=2e-3, atol=3e-4)
        discrepancies.append(float(np.max(np.abs(batch-direct))))
    dump(a.output/f'numerical_audit_{it}.json', dict(jvp_equals_reverse_gradient=True,
        batch_equals_single=True, max_errors=discrepancies, direction_norms=[norm1,norm2]))
    print('numerical_audit_passed', it, 'elapsed', perf_counter()-t0, flush=True)
    if a.audit_only:
        continue
    path = a.output/f'scores_{it:011d}.h5'
    with h5py.File(path, 'a') as f:
        identity = json.dumps(provenance, sort_keys=True)
        if 'icl_score' not in f:
            for n, dtype in [('train_loss','f4'),('icl_score','f4'),('ciwl_score','f4'),
                             ('relative_icl','f4'),('label','i1'),('both_positive','?')]:
                f.create_dataset(n, shape=(a.limit,), dtype=dtype, chunks=(4096,))
            f.attrs['provenance'] = identity
            f.attrs['completed_sequences'] = 0
            f.attrs['labels'] = '0=reject,1=ICL_tendency,2=CIWL_tendency; no causal labels'
        assert f.attrs['provenance'] == identity
        done = int(f.attrs['completed_sequences'])
        while done < a.limit:
            if perf_counter()-t0 > 9000:
                f.attrs['completed_sequences'] = done
                f.flush()
                raise RuntimeError('Safe scoring boundary at 2.5h; retain progress for compute-route review')
            end = min(done+32, a.limit)
            x, y = samples(np.arange(done, end))
            arrays = [np.asarray(v) for v in score(model, d1, d2, x, y)]
            assert all(np.isfinite(v).all() for v in arrays)
            l, s1, s2 = arrays
            diff, labels = classify(s1,s2)
            values = [l,s1,s2,0.5+0.5*diff,labels,(s1>0)&(s2>0)]
            for n,v in zip(['train_loss','icl_score','ciwl_score','relative_icl','label','both_positive'],values):
                f[n][done:end] = v
            done = end
            if done%8192 == 0 or done == a.limit:
                f.attrs['completed_sequences'] = done
                f.flush()
            if done%65536 == 0 or done == a.limit:
                dump(a.output/'progress.json',dict(stage='scoring',checkpoint=it,completed_sequences=done,
                    total_sequences=a.limit,elapsed_seconds=perf_counter()-t0))
                print('scored',it,done,'elapsed',round(perf_counter()-t0,2),flush=True)
        s1, s2, labels = [np.asarray(f[n]) for n in ['icl_score','ciwl_score','label']]
        summaries[str(it)] = dict(counts={str(k):int((labels==k).sum()) for k in [0,1,2]},
            both_positive=int(((s1>0)&(s2>0)).sum()), exact_match_count=int(exact.sum()),
            exact_match_by_label={str(k):int((exact&(labels==k)).sum()) for k in [0,1,2]},
            quantiles={n:np.quantile(s,[0,.01,.1,.5,.9,.99,1]).tolist() for n,s in [('icl',s1),('ciwl',s2)]})
    dump(a.output/'classification_summary.json', summaries)
if a.audit_only:
    print('audit_only_complete',flush=True)
    sys.exit(0)

# Actual Adam and held-out metrics use the original baseline's precision policy.
jax.config.update('jax_default_matmul_precision', original_matmul_precision)

def evaluate_holdout(model):
    result = {}
    for name, ev in evaluators.items():
        rows = []
        for i in range(2500,5000,32):
            metrics = main.eval_step(model=model, fwd_fn=fwd,
                x=ev['examples'][i:i+32],y=ev['labels'][i:i+32],key=key)
            rows.append({n:np.asarray(metrics[n]) for n in ['loss','acc','in_context_acc']})
        result[name] = {n:float(np.concatenate([r[n] for r in rows]).mean()) for n in rows[0]}
    return result

with h5py.File(a.output/'scores_00004400000.h5','r') as f:
    s1,s2 = [np.asarray(f[n]) for n in ['icl_score','ciwl_score']]
diff, labels = classify(s1,s2)
pools = {}
for label,name,descending in [(1,'icl',True),(2,'ciwl',False)]:
    ids = np.flatnonzero(labels == label)
    ids = ids[np.argsort(diff[ids])]
    n = max(4096, int(len(ids)*0.1))
    assert len(ids) >= 4096, 'Not enough confidently positive candidates; validation inconclusive'
    pools[name] = ids[-n:] if descending else ids[:n]
ck = load(4400000)
before = evaluate_holdout(ck['model'])
validation_path = a.output/'validation.json'
validation = json.loads(validation_path.read_text()) if validation_path.exists() else dict(
    initial=before, runs=[], sequences_per_group=4096, selection_seeds=[103,104],
    status='running',scope='exploratory group-level Adam test, not per-row causal proof')
selection_file = a.output/'validation_ids.npz'
all_ids = {}
for seed in [103,104]:
    rng = np.random.default_rng(seed)
    for name in ['icl','ciwl','random']:
        pool = pools[name] if name in pools else np.arange(a.limit)
        all_ids[f'{seed}_{name}'] = rng.choice(pool,4096,replace=False)
np.savez_compressed(selection_file, **all_ids)
for seed in [103,104]:
    for name in ['icl','ciwl','random']:
        if any(r['seed']==seed and r['group']==name for r in validation['runs']):
            continue
        ids = all_ids[f'{seed}_{name}']
        model, state, mkey = ck['model'], ck['opt_state'], ck['seeds']['train_model_seed']
        for i in range(0,4096,32):
            mkey,current = jax.random.split(mkey)
            x,y = samples(ids[i:i+32])
            metrics,model,state = main.train_step(model=model,fwd_fn=fwd,optimizer=optimizer,
                opt_state=state,microbs=32,weight_decay=0.,x=x,y=y,key=current)
        after = evaluate_holdout(model)
        save = dict(iter=4400000+4096,model=model,opt_state=state,
                    seeds={**ck['seeds'],'train_model_seed':mkey})
        target = a.output/f'validation_{seed}_{name}.eqx'
        eqx.tree_serialise_leaves(target,save)
        # These are selected-data branches, not original-stream resume points.
        run = dict(seed=seed,group=name,after=after,
            loss_reduction={n:before[n]['loss']-after[n]['loss'] for n in after},
            exact_match_fraction=float(exact[ids].mean()),unique_query_classes=int(len(np.unique(classes[ids,2]))),
            checkpoint_sha256=sha(target),checkpoint_semantics='selected-data branch; train_data_seed remains original anchor; resume using validation_ids.npz')
        validation['runs'].append(run)
        dump(validation_path,validation)
        print('validation',seed,name,json.dumps(run['loss_reduction']),flush=True)
gate = []
for seed in [103,104]:
    bygroup = {r['group']:r for r in validation['runs'] if r['seed']==seed}
    gate.append(dict(seed=seed,icl_advantage=bygroup['icl']['loss_reduction']['icl']-bygroup['random']['loss_reduction']['icl'],
        ciwl_advantage=bygroup['ciwl']['loss_reduction']['iwl_copy_avail']-bygroup['random']['loss_reduction']['iwl_copy_avail']))
validation.update(status='completed',gate=gate,scale_gate_passed=all(r['icl_advantage']>0 and r['ciwl_advantage']>0 for r in gate),
    expansion='not started; review pilot before any 10M scoring',elapsed_seconds=perf_counter()-t0)
dump(validation_path,validation)

# Balanced small export for the language-model review in the current conversation.
review_ids = np.unique(np.concatenate([np.argsort(diff)[:16],np.argsort(diff)[-16:],
    np.argsort(np.abs(diff))[:16],np.flatnonzero((s1>0)&(s2>0))[:16]]))
stage_labels = []
for it in a.checkpoints:
    with h5py.File(a.output/f'scores_{it:011d}.h5','r') as f:
        stage_labels.append(np.asarray(f['label']))
stage_labels = np.stack(stage_labels)
conflict = np.flatnonzero((stage_labels==1).any(0)&(stage_labels==2).any(0))
review_ids = np.unique(np.concatenate([review_ids, conflict[:16]]))
review = []
for i in review_ids:
    review.append(dict(sequence_id=int(i),class_idx=classes[i].tolist(),exemplar_idx=exemplars[i].tolist(),
        exact_match=bool(exact[i]),icl_score=float(s1[i]),ciwl_score=float(s2[i]),
        relative_icl=float(.5+.5*diff[i]),stage_labels=stage_labels[:,i].tolist()))
dump(a.output/'llm_review_samples.json',dict(checkpoints=a.checkpoints,not_probability=True,samples=review,
    review_questions=['Do labels and exact-match fields obey original rules?',
        'Are joint-positive or ambiguous rows being mistaken for exclusive causal effects?',
        'Does held-out Adam validation support the proposed ranking?']))
dump(a.output/'progress.json',dict(stage='completed_pilot',elapsed_seconds=perf_counter()-t0,
    scale_gate_passed=validation['scale_gate_passed'],stage_conflict_count=int(len(conflict))))
print('completed_pilot',json.dumps(validation['gate']),flush=True)
