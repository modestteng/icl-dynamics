"""Read metrics already saved by the unchanged author CLI; no model calls."""
from pathlib import Path
import json, pickle, csv
import numpy as np
import h5py

w = Path(__file__).resolve().parent
plan = json.loads((w / 'plan.json').read_text())
expected = json.loads((w / 'expected_baseline.json').read_text())
rows = []
with h5py.File(w / 'outputs/per_sequence_metrics.h5', 'a') as h:
    for job in plan:
        jid = job['job_id']
        if not (w / 'logs' / (jid + '.done')).exists():
            continue
        files = list((w / 'runs' / jid / 'plots').glob('*.pkl'))
        assert len(files) == 1, jid
        with files[0].open('rb') as f:
            info = pickle.load(f)['info']
        assert info['iters'] == [job['stage']]
        arrays = {k: np.asarray(info[k][0]) for k in
                  ['in_context_acc', 'in_context_prob', 'in_context_loss', 'loss', 'acc', 'prob']}
        assert all(a.shape == (5000,) and np.isfinite(a).all() for a in arrays.values()), jid
        if jid not in h:
            g = h.create_group(jid)
            for k, a in arrays.items():
                g.create_dataset(k, data=a, compression='gzip')
            g.attrs['in_context_metric_valid'] = job['evaluator'] != 'pure_iwl'
        else:
            for k, a in arrays.items():
                np.testing.assert_array_equal(h[jid][k][:], a)
        valid = job['evaluator'] != 'pure_iwl'
        row = dict(stage=job['stage'], condition=job['condition'], evaluator=job['evaluator'],
                   count=5000, in_context_correct_count=int(arrays['in_context_acc'].sum()) if valid else None,
                   in_context_acc=float(arrays['in_context_acc'].mean(dtype=np.float64)) if valid else None,
                   full_correct_count=int(arrays['acc'].sum()),
                   full_acc=float(arrays['acc'].mean(dtype=np.float64)),
                   in_context_prob=float(arrays['in_context_prob'].mean(dtype=np.float64)) if valid else None,
                   in_context_loss=float(arrays['in_context_loss'].mean(dtype=np.float64)) if valid else None,
                   full_prob=float(arrays['prob'].mean(dtype=np.float64)),
                   full_loss=float(arrays['loss'].mean(dtype=np.float64)))
        rows.append(row)
        if job['condition'] == 'baseline':
            metric = row['in_context_acc'] if valid else row['full_acc']
            ref = expected['baseline_logged_accuracy_by_stage'][str(job['stage'])][job['evaluator']]
            row['baseline_log_difference'] = metric - ref
    h.flush()
if rows:
    fields = list(rows[0]) + ['baseline_log_difference']
    with (w / 'outputs/metrics.tsv').open('w') as f:
        wr = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        wr.writeheader(); wr.writerows(rows)
    (w / 'outputs/summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows[-1]), flush=True)
print('completed_jobs', len(rows), 'of', len(plan), flush=True)
