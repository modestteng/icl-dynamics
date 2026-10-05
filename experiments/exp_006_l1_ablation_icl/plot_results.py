"""Show new ICL and frozen reference CIWL results for the same interventions."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent / 'outputs'
s = json.loads((out / 'summary.json').read_text())
conditions = ['baseline', 'preserve_qkv', 'recompute_k', 'recompute_v', 'recompute_q', 'ablate_l1']
labels = ['Original model', 'Ablate L1; preserve L2 Q, K, V',
          'Ablate L1; recompute L2 K', 'Ablate L1; recompute L2 V',
          'Ablate L1; recompute L2 Q', 'Ablate L1; recompute all L2 Q, K, V']
icl = [100 * s['results'][c]['icl']['in_context_acc'] for c in conditions]
ciwl = [100 * s['reference_exp005_results'][c]['iwl_copy_avail']['in_context_acc'] for c in conditions]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
fig, ax = plt.subplots(figsize=(12, 6.7), layout='constrained')
pos = np.arange(6)
ax.barh(pos - .18, icl, height=.32, color='#0072B2', label='ICL: newly evaluated')
ax.barh(pos + .18, ciwl, height=.32, color='#009E73', label='CIWL: recorded exp005 reference')
ax.set_yticks(pos, labels)
ax.invert_yaxis()
ax.set_xlim(0, 110)
ax.set_xlabel('In-context accuracy (%)')
ax.axvline(50, color='#6b7280', linestyle='--', linewidth=1.1, label='Chance: 50%')
for offset, values in [(-.18, icl), (.18, ciwl)]:
    for p, v in zip(pos, values):
        ax.text(v + .5, p + offset, f'{v:.2f}%', ha='left', va='center', fontsize=11)
ax.grid(axis='x', alpha=.15)
ax.set_axisbelow(True)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.10), ncol=2, fontsize=10)
fig.suptitle('ICL after the same L1 / L2-QKV ablations\nSame saved 20M checkpoint; original 5,000 sequences per evaluator', fontsize=16)
fig.savefig(out / 'icl_ciwl_ablations.png', dpi=190, bbox_inches='tight')
fig.savefig(out / 'icl_ciwl_ablations.svg', bbox_inches='tight')
plt.close(fig)
rows = ['condition\ticl_accuracy\ticl_delta_vs_baseline\tciwl_accuracy_reference\tflip_accuracy_reference\ticl_context_loss\ticl_context_probability']
for c in conditions:
    r = s['results'][c]['icl']
    old = s['reference_exp005_results'][c]
    rows.append('\t'.join(str(v) for v in [c, r['in_context_acc'],
        s['paired_vs_baseline'][c]['icl']['accuracy_delta'],
        old['iwl_copy_avail']['in_context_acc'], old['flip_icl']['in_context_acc'],
        r['in_context_loss'], r['in_context_prob']]))
(out / 'comparison.tsv').write_text('\n'.join(rows) + '\n')
print('plots_completed', flush=True)
