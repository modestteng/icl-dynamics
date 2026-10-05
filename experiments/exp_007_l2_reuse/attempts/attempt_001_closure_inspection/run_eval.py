"""Minimal Figure3b evaluation through the unchanged author's graft-out path."""
from pathlib import Path
from copy import deepcopy
from time import perf_counter
import argparse
import csv
import hashlib
import json
import sys

work = Path(__file__).resolve().parent
root = Path.home() / 'research/icl-dynamics'
source = root / 'exp_001_hardware_check/source'
baseline_root = root / 'exp_002_main_reproduction'
sys.path.insert(0, str(source))
sys.path.insert(1, str(work / 'author_reference'))
import numpy as np
import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import main
import main_utils
import opto
from forward_metrics import make_forward_fn_ic_only

t0 = perf_counter()
out = work / 'outputs'
out.mkdir(exist_ok=True)
assert jax.default_backend() == 'gpu'
assert jax.config.jax_default_matmul_precision is None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


e = json.loads((work / 'expected_baseline.json').read_text())
cfg = baseline_root / 'runs/main/config.json'
evfile = baseline_root / 'runs/default_eval/eval_data.h5'
assert sha(cfg) == e['configuration_sha256']
assert sha(evfile) == e['evaluation_data_sha256']
assert sha(work / 'reference_baseline/learning_curves.tsv') == e['baseline_curve_sha256']
assert sha(work / 'author_reference/forward_metrics.py') == e['author_forward_sha256']
for n, h in e['source_sha256'].items():
    assert sha(source / n) == h, n
stages = e['stages']
assert stages == [0, 2000000, 4400000, 8000000, 20000000]


def path(stage):
    run = 'main' if stage < 2950016 else 'main_resume_00002950016'
    return baseline_root / f'runs/{run}/checkpoints/{stage:011d}.eqx'


for t in stages:
    assert path(t).stat().st_size == e['checkpoint_bytes']
    assert sha(path(t)) == e['checkpoint_sha256_by_stage'][str(t)], t
opts = main_utils.get_opts_from_json_file(cfg)
model0 = main_utils.get_model_from_opts(opts, (512,))
optimizer = main_utils.get_optimizer_from_opts(opts)
key = jax.random.PRNGKey(0)
template = dict(iter=-1, model=model0,
    opt_state=optimizer.init(eqx.filter(model0, eqx.is_array)),
    seeds=dict(eval_model_seed=key, train_data_seed=key, train_model_seed=key))


def restore(t):
    ck = eqx.tree_deserialise_leaves(path(t), template)
    assert int(ck['iter']) == t and int(ck['opt_state'][0].count) == t // 32
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(ck)
               if eqx.is_array(v))
    return ck['model']


donor = restore(20000000)
parser = argparse.ArgumentParser()
opto.add_args_to_parser(parser)
graft_opts = parser.parse_args([])
graft_opts.opto_graft_out_model_ckpt = str(path(20000000))
graft_opts.opto_graft_out_model_cfg = str(cfg)
graft_opts.opto_graft_out_model_from_layer = 0
fwd_train = opto.make_fn_from_opts(opts)
graft_call = opto.make_fn_from_opts(deepcopy(graft_opts), default_fn=fwd_train)
closure = {name: cell.cell_contents for name, cell in
           zip(graft_call.__code__.co_freevars, graft_call.__closure__)}
closed_donor = closure['graft_out_model']


def same_arrays(a, b):
    aa = [v for v in jax.tree_util.tree_leaves(a) if eqx.is_array(v)]
    bb = [v for v in jax.tree_util.tree_leaves(b) if eqx.is_array(v)]
    assert len(aa) == len(bb)
    for x, y in zip(aa, bb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    return len(aa)


donor_leaf_count = same_arrays(closed_donor, donor)
assert not any(getattr(graft_opts, name) for name in
    ['opto_preserve_patterns', 'opto_preserve_queries', 'opto_preserve_keys',
     'opto_preserve_values', 'opto_ablate_heads'])
graft_fn = main.make_batched_fn(make_forward_fn_ic_only(deepcopy(graft_opts),
                               default_fn=fwd_train), 1024)
native_fn = main.make_batched_fn(make_forward_fn_ic_only(parser.parse_args([]),
                                default_fn=fwd_train), 1024)
sets = ['train_eval', 'icl', 'iwl_copy_avail', 'flip_icl']
data = {}
with h5py.File(evfile, 'r') as f:
    for n in sets:
        x = jnp.array(f[n]['examples'])
        y = jnp.array(f[n]['labels'])
        assert x.shape == (5000, 3, 512) and y.shape == (5000, 3)
        assert np.isfinite(np.asarray(x)).all()
        assert np.all(np.asarray((y[:, :2] == y[:, -1, None]).sum(1)) == 1)
        if n == 'icl':
            assert np.all(np.isin(np.asarray(y), [0, 1]))
        data[n] = (x, y)
baseline_rows = list(csv.DictReader((work / 'reference_baseline/learning_curves.tsv').open(), delimiter='\t'))
baseline = {int(float(r['training_sequences'])): r for r in baseline_rows}
cols = dict(train_eval='train_in_context_acc', icl='icl_in_context_acc',
            iwl_copy_avail='ciwl_in_context_acc', flip_icl='flip_in_context_acc')
native_reference = {str(t): {n: float(baseline[t][col]) for n, col in cols.items()}
                    for t in stages}
prov = dict(stages=stages, fixed_donor_iter=20000000,
    checkpoint_sha256=e['checkpoint_sha256_by_stage'],
    config_sha256=sha(cfg), evaluation_sha256=sha(evfile),
    source_sha256=e['source_sha256'], baseline_curve_sha256=e['baseline_curve_sha256'],
    script_sha256=sha(Path(__file__)), protocol_sha256=sha(work / 'protocol.md'),
    plot_sha256=sha(work / 'plot_results.py'), author_forward_sha256=e['author_forward_sha256'],
    author_figure3b_cells_sha256=sha(work / 'author_reference/figure3b_cells.json'),
    graft_flags=vars(graft_opts), batch_size=1024, eval_key=[0, 0],
    variable_first_half='source checkpoint input embeddings and L1',
    fixed_second_half='donor L2, final norm and unembedding',
    preserved_qkv_activations=False, evaluators=sets, sequences_per_evaluator=5000,
    precision='float32, original default matmul precision',
    jax=jax.__version__, equinox=eqx.__version__, devices=[str(d) for d in jax.devices()],
    donor_parameter_arrays_verified=donor_leaf_count, no_training=True)
(out / 'provenance.json').write_text(json.dumps(prov, indent=2) + '\n')


def traced_boundary(m, x, y, k):
    class Capture:
        def __init__(self, native):
            self.native = native
            self.transformer = native.transformer
            self.calls = []

        def call_with_all_aux(self, **kwargs):
            r = self.native.call_with_all_aux(**kwargs)
            self.calls.append(r)
            return r

    capture = Capture(m)
    r = graft_call(capture, x, y, k)
    assert len(capture.calls) == 1
    source_boundary = capture.calls[0]['transformer_output']['block_outputs'][0]['out']
    blocks = r['transformer_output']['block_outputs']
    return dict(source_boundary=source_boundary, injected_boundary=blocks[0]['out'],
        second_layer_input=blocks[1]['x'],
        q=blocks[1]['attn_output']['q'], k=blocks[1]['attn_output']['k'],
        v=blocks[1]['attn_output']['v'])


boundary_fn = eqx.filter_jit(traced_boundary)
hp = out / 'per_sequence_metrics.h5'
results = {}
audit = dict(donor_parameter_arrays_verified=donor_leaf_count,
             no_qkv_or_attention_pattern_preservation=True,
             sample='original fixed ICL row0', stages={})
with h5py.File(hp, 'a') as f:
    if 'provenance' not in f.attrs:
        f.attrs['provenance'] = json.dumps(prov, sort_keys=True)
    assert json.loads(f.attrs['provenance']) == prov

    def evaluate(group, model, fn, evaluator):
        if group in f and f[group].attrs.get('completed', False):
            return json.loads(f[group].attrs['summary'])
        if group in f:
            del f[group]
        x, y = data[evaluator]
        metrics = {n: np.asarray(v) for n, v in fn(model, x, y, key=key).items()}
        assert all(v.shape == (5000,) and np.isfinite(v).all() for v in metrics.values())
        assert set(metrics) == {'in_context_acc', 'in_context_prob', 'in_context_loss', 'loss'}
        r = {n: float(v.mean(dtype=np.float64)) for n, v in metrics.items()}
        r.update(correct_count=int(metrics['in_context_acc'].sum()), count=5000)
        g = f.create_group(group)
        for n, v in metrics.items():
            g.create_dataset(n, data=v, compression='gzip')
        g.attrs['summary'] = json.dumps(r)
        g.attrs['completed'] = True
        f.flush()
        print('evaluated', group, 'acc', r['in_context_acc'],
              'elapsed', round(perf_counter() - t0, 2), flush=True)
        return r

    for t in stages:
        model = restore(t)
        if t == 20000000:
            same_arrays(model, donor)
        x, y = (a[0] for a in data['icl'])
        tensors = {n: np.asarray(v) for n, v in boundary_fn(model, x, y, key).items()}
        np.testing.assert_array_equal(tensors['source_boundary'], tensors['injected_boundary'])
        np.testing.assert_array_equal(tensors['source_boundary'], tensors['second_layer_input'])
        assert all(np.isfinite(v).all() for v in tensors.values())
        gname = f'audit/{t:011d}'
        if gname in f:
            del f[gname]
        g = f.create_group(gname)
        for n, v in tensors.items():
            g.create_dataset(n, data=v)
        audit['stages'][str(t)] = dict(boundary_max_abs_difference=0.,
            l2_input_max_abs_difference=0., source_iter=t)
        results[str(t)] = {n: evaluate(f'graft/{t:011d}/{n}', model, graft_fn, n) for n in sets}
        (out / 'progress.json').write_text(json.dumps(results, indent=2) + '\n')
    native20 = {n: evaluate(f'native20m/{n}', donor, native_fn, n) for n in sets}
    identity = {}
    for n in sets:
        original = np.asarray(f[f'native20m/{n}/in_context_acc'], dtype=np.float64)
        grafted = np.asarray(f[f'graft/00020000000/{n}/in_context_acc'], dtype=np.float64)
        identity[n] = dict(native20m=native20[n]['in_context_acc'],
            original_logged=native_reference['20000000'][n],
            graft20m=results['20000000'][n]['in_context_acc'],
            individual_prediction_disagreements=int(np.sum(original != grafted)),
            native_minus_logged=float(original.mean() - native_reference['20000000'][n]),
            graft_minus_native=float(grafted.mean() - original.mean()))
    ref_qkv = {n: np.asarray(f[f'audit/00020000000/{n}']) for n in ['q', 'k', 'v']}
    for t in stages:
        audit['stages'][str(t)]['l2_qkv_change_from_20m_self_graft'] = {
            n: float(np.max(np.abs(np.asarray(f[f'audit/{t:011d}/{n}']) - ref_qkv[n])))
            for n in ref_qkv}
same_arrays(closed_donor, donor)
for t in stages:
    assert sha(path(t)) == e['checkpoint_sha256_by_stage'][str(t)]
assert sha(cfg) == e['configuration_sha256'] and sha(evfile) == e['evaluation_data_sha256']
audit['passed'] = True
(out / 'graft_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
differences = {str(t): {n: results[str(t)][n]['in_context_acc'] - native_reference[str(t)][n]
                       for n in sets} for t in stages}
summary = dict(stages=stages, results=results, original_baseline=native_reference,
    graft_minus_original=differences, native20m_control=native20, self_graft_identity=identity,
    descriptive_checks=dict(
        icl_above_chance_at_original_peak=results['4400000']['icl']['in_context_acc'] > .5,
        icl_lower_at_20m_than_4_4m=results['20000000']['icl']['in_context_acc'] < results['4400000']['icl']['in_context_acc'],
        ciwl_higher_at_20m_than_4_4m=results['20000000']['iwl_copy_avail']['in_context_acc'] > results['4400000']['iwl_copy_avail']['in_context_acc']),
    scope='Five prespecified representative checkpoints, not a dense full-trajectory reproduction',
    elapsed_seconds=perf_counter() - t0, no_training=True)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
rows = ['first_half_sequences\tevaluator\tgraft_accuracy\toriginal_accuracy\tdelta\tcorrect_count\tcontext_loss\tcontext_probability\tloss']
for t in stages:
    for n in sets:
        r = results[str(t)][n]
        rows.append('\t'.join(str(v) for v in [t, n, r['in_context_acc'],
            native_reference[str(t)][n], differences[str(t)][n], r['correct_count'],
            r['in_context_loss'], r['in_context_prob'], r['loss']]))
(out / 'metrics.tsv').write_text('\n'.join(rows) + '\n')
(out / 'finished.json').write_text(json.dumps(dict(status='completed',
    elapsed_seconds=perf_counter() - t0, graft_stages=5, evaluators=4,
    graft_evaluation_sequences=100000, native_control_sequences=20000,
    per_sequence_metrics_sha256=sha(hp), baseline_inputs_unchanged=True), indent=2) + '\n')
print('completed', round(perf_counter() - t0, 2), flush=True)
