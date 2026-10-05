"""Minimal adapter of the author's Section 4 notebook ablations; no training."""
from pathlib import Path
from copy import deepcopy
from time import perf_counter
import argparse
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

start = perf_counter()
outdir = work / 'outputs'
outdir.mkdir(exist_ok=True)
assert jax.default_backend() == 'gpu'
assert jax.config.jax_default_matmul_precision is None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


expected = json.loads((work / 'expected_baseline.json').read_text())
ckpath = Path(expected['checkpoint'])
cfgpath = baseline_root / 'runs/main/config.json'
evalpath = baseline_root / 'runs/default_eval/eval_data.h5'
assert ckpath.stat().st_size == expected['checkpoint_bytes']
assert sha(ckpath) == expected['checkpoint_sha256']
assert sha(cfgpath) == expected['configuration_sha256']
assert sha(evalpath) == expected['evaluation_data_sha256']
for name, digest in expected['source_sha256'].items():
    assert sha(source / name) == digest, name

opts = main_utils.get_opts_from_json_file(cfgpath)
model0 = main_utils.get_model_from_opts(opts, (512,))
optimizer = main_utils.get_optimizer_from_opts(opts)
key = jax.random.PRNGKey(0)
template = dict(iter=-1, model=model0,
                opt_state=optimizer.init(eqx.filter(model0, eqx.is_array)),
                seeds=dict(eval_model_seed=key, train_data_seed=key, train_model_seed=key))
ck = eqx.tree_deserialise_leaves(ckpath, template)
assert int(ck['iter']) == 20000000
assert int(ck['opt_state'][0].count) == 625000
assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(ck)
           if eqx.is_array(v))
model = ck['model']
assert len(model.transformer.blocks) == 2
assert model.transformer.blocks[0].attn.num_heads == 8

parser = argparse.ArgumentParser()
opto.add_args_to_parser(parser)
base_opts = parser.parse_args([])
fwd_from_train = opto.make_fn_from_opts(opts)
preservations = {
    'baseline': None,
    'preserve_qkv': ['q', 'k', 'v'],
    'recompute_k': ['q', 'v'],
    'recompute_v': ['q', 'k'],
    'recompute_q': ['k', 'v'],
    'ablate_l1': [],
}
flag_names = {'q': 'opto_preserve_queries', 'k': 'opto_preserve_keys',
              'v': 'opto_preserve_values'}
condition_opts = {}
for name, preserve in preservations.items():
    condition_opts[name] = deepcopy(base_opts)
    if preserve is not None:
        condition_opts[name].opto_ablate_heads = [f'0:{h}' for h in range(8)]
        for quantity in preserve:
            setattr(condition_opts[name], flag_names[quantity], True)

sets = ['train_eval', 'iwl_copy_avail', 'flip_icl']
data = {}
with h5py.File(evalpath, 'r') as f:
    for name in sets:
        x = jnp.array(f[name]['examples'])
        y = jnp.array(f[name]['labels'])
        assert x.shape == (5000, 3, 512) and y.shape == (5000, 3)
        assert np.isfinite(np.asarray(x)).all()
        assert np.all(np.asarray((y[:, :2] == y[:, -1, None]).sum(1)) == 1)
        data[name] = (x, y)

provenance = dict(
    checkpoint=str(ckpath), checkpoint_sha256=sha(ckpath),
    checkpoint_iter=int(ck['iter']), adam_count=int(ck['opt_state'][0].count),
    config_sha256=sha(cfgpath), evaluation_sha256=sha(evalpath),
    source_sha256=expected['source_sha256'],
    analysis_sha256=sha(Path(__file__)),
    author_function_sha256=sha(work / 'author_reference/forward_metrics.py'),
    author_notebook_cells_sha256=sha(work / 'author_reference/notebook_cells.json'),
    protocol_sha256=sha(work / 'protocol.md'),
    plot_code_sha256=sha(work / 'plot_results.py'),
    conditions={n: vars(o) for n, o in condition_opts.items()},
    evaluators=sets, samples_per_evaluator=5000, eval_batch_size=1024,
    eval_key=[0, 0], precision='float32, original default matmul precision',
    devices=[str(d) for d in jax.devices()], jax=jax.__version__,
    equinox=eqx.__version__, numpy=np.__version__,
    scope='Section4 L1 attention-head ablation on one existing20M model; no training',
)
(outdir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

# Validate actual preserved/recomputed tensors rather than only checking flags.
audit_path = outdir / 'intervention_audit.json'
if audit_path.exists():
    audit = json.loads(audit_path.read_text())
    assert audit['provenance'] == provenance
    assert audit['passed']
else:
    x0, y0 = (a[0] for a in data['iwl_copy_avail'])
    original = eqx.filter_jit(fwd_from_train)(model, x0, y0, key)
    original_l2 = original['transformer_output']['block_outputs'][1]['attn_output']
    audit = dict(provenance=provenance, sample='iwl_copy_avail row0',
                 rtol=2e-5, atol=3e-6, conditions={})
    for name, preserve in preservations.items():
        call = opto.make_fn_from_opts(deepcopy(condition_opts[name]), default_fn=fwd_from_train)
        r = eqx.filter_jit(call)(model, x0, y0, key)
        blocks = r['transformer_output']['block_outputs']
        if preserve is not None:
            assert np.max(np.abs(np.asarray(blocks[0]['attn_output']['v']))) == 0
        values = {}
        for quantity in ['q', 'k', 'v']:
            actual = np.asarray(blocks[1]['attn_output'][quantity])
            ref = np.asarray(original_l2[quantity])
            difference = float(np.max(np.abs(actual - ref)))
            retained = preserve is None or quantity in preserve
            if retained:
                np.testing.assert_allclose(actual, ref, rtol=2e-5, atol=3e-6)
            else:
                assert difference > 1e-5, (name, quantity)
            values[quantity] = dict(preserved=retained, max_abs_change=difference)
        audit['conditions'][name] = values
        print('intervention_audit', name, values, flush=True)
    audit['passed'] = True
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')

result_path = outdir / 'per_sequence_metrics.h5'
results = {}
with h5py.File(result_path, 'a') as f:
    if 'provenance' not in f.attrs:
        f.attrs['provenance'] = json.dumps(provenance, sort_keys=True)
    assert json.loads(f.attrs['provenance']) == provenance
    for condition in preservations:
        results[condition] = {}
        fn = main.make_batched_fn(
            make_forward_fn_ic_only(deepcopy(condition_opts[condition]), default_fn=fwd_from_train),
            1024)
        for evaluator in sets:
            group = f'{condition}/{evaluator}'
            if group in f and f[group].attrs.get('completed', False):
                results[condition][evaluator] = json.loads(f[group].attrs['summary'])
                continue
            if group in f:
                del f[group]
            x, y = data[evaluator]
            metrics = {k: np.asarray(v) for k, v in fn(model, x, y, key=key).items()}
            assert all(v.shape == (5000,) and np.isfinite(v).all() for v in metrics.values())
            assert set(metrics) == {'in_context_loss', 'in_context_prob', 'in_context_acc', 'loss'}
            assert np.all((metrics['in_context_prob'] >= 0) & (metrics['in_context_prob'] <= 1))
            mean = {k: float(v.mean(dtype=np.float64)) for k, v in metrics.items()}
            mean['correct_count'] = int(metrics['in_context_acc'].sum())
            mean['count'] = 5000
            g = f.create_group(group)
            for k, v in metrics.items():
                g.create_dataset(k, data=v, compression='gzip')
            g.attrs['summary'] = json.dumps(mean)
            g.attrs['completed'] = True
            f.flush()
            results[condition][evaluator] = mean
            print('evaluated', condition, evaluator, 'in_context_acc', mean['in_context_acc'],
                  'elapsed', round(perf_counter() - start, 2), flush=True)
            (outdir / 'progress.json').write_text(json.dumps(results, indent=2) + '\n')
    paired = {}
    for condition in preservations:
        paired[condition] = {}
        for evaluator in sets:
            base = np.asarray(f[f'baseline/{evaluator}/in_context_acc'], dtype=np.float64)
            actual = np.asarray(f[f'{condition}/{evaluator}/in_context_acc'], dtype=np.float64)
            delta = actual - base
            m = float(delta.mean())
            se = float(delta.std(ddof=1) / np.sqrt(5000))
            paired[condition][evaluator] = dict(accuracy_delta=m,
                normal95_interval=[m - 1.96 * se, m + 1.96 * se],
                correct_to_wrong=int(np.sum((base == 1) & (actual == 0))),
                wrong_to_correct=int(np.sum((base == 0) & (actual == 1))))

summary = dict(results=results, paired_vs_baseline=paired,
    original_logged_baseline=expected['baseline_logged_accuracy'],
    baseline_difference_vs_original={n: results['baseline'][n]['in_context_acc'] - v
                                    for n, v in expected['baseline_logged_accuracy'].items()},
    paper_ciwl_accuracy=dict(baseline=0.987, preserve_qkv=0.985,
                            recompute_k=0.575, recompute_v=0.593, recompute_q=0.954),
    comparison_scope='Author reported64M checkpoint vs our saved20M model; qualitative replication',
    elapsed_seconds=perf_counter() - start,
    uncertainty='Paired intervals are evaluation-sample variation conditional on one fixed model',
    no_training=True)
(outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
rows = ['condition\tevaluator\tin_context_acc\tcorrect_count\tin_context_loss\tin_context_prob\tloss\tdelta_vs_baseline']
for c, evaluations in results.items():
    for n, r in evaluations.items():
        rows.append('\t'.join(str(x) for x in [c, n, r['in_context_acc'], r['correct_count'],
            r['in_context_loss'], r['in_context_prob'], r['loss'], paired[c][n]['accuracy_delta']]))
(outdir / 'metrics.tsv').write_text('\n'.join(rows) + '\n')
assert sha(ckpath) == expected['checkpoint_sha256']
assert sha(cfgpath) == expected['configuration_sha256']
assert sha(evalpath) == expected['evaluation_data_sha256']
(outdir / 'finished.json').write_text(json.dumps(dict(status='completed',
    elapsed_seconds=perf_counter() - start, conditions=6, evaluators=3,
    per_sequence_metrics_sha256=sha(result_path), baseline_files_unchanged=True), indent=2) + '\n')
print('completed', round(perf_counter() - start, 2), flush=True)
