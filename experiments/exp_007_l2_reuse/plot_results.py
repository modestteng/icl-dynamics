"""Figure3b core comparison: all original points, five measured graft points."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

work = Path(__file__).resolve().parent
out = work / 'outputs'
s = json.loads((out / 'summary.json').read_text())
ref = np.genfromtxt(work / 'reference_baseline/learning_curves.tsv', names=True, delimiter='\t')
evaluators = [('train_eval', 'Train', 'train_in_context_acc', '#4b5563'),
              ('icl', 'ICL', 'icl_in_context_acc', '#0072B2'),
              ('iwl_copy_avail', 'CIWL', 'ciwl_in_context_acc', '#009E73'),
              ('flip_icl', 'Flip', 'flip_in_context_acc', '#D55E00')]
x = np.array(s['stages']) / 1e6
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, layout='constrained')
for ax, (name, label, col, color) in zip(axes.flat, evaluators):
    ax.plot(ref['training_sequences'] / 1e6, ref[col], color=color, alpha=.35,
            lw=2, label='Original trained model (recorded)')
    y = [s['results'][str(t)][name]['in_context_acc'] for t in s['stages']]
    ax.plot(x, y, 'o--', color=color, lw=2, ms=7,
            label='Fixed 20M L2 + readout; varying first half')
    ax.axhline(.5, color='gray', ls=':', lw=1)
    ax.set_title(label)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('First-half training sequences (million)')
    ax.set_ylabel('In-context accuracy')
    ax.grid(alpha=.15)
axes[0, 0].legend(fontsize=9, loc='lower right')
fig.suptitle('Figure 3b: reuse the saved 20M second half\nFive measured graft checkpoints; dashed links guide the eye only', fontsize=16)
fig.savefig(out / 'figure3b_l2_reuse.png', dpi=190, bbox_inches='tight')
fig.savefig(out / 'figure3b_l2_reuse.svg', bbox_inches='tight')
plt.close(fig)
print('plots_completed', flush=True)
