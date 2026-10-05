"""Plot measured CIWL accuracy at the saved baseline checkpoint."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

outdir = Path(__file__).resolve().parent / 'outputs'
s = json.loads((outdir / 'summary.json').read_text())
conditions = ['baseline', 'preserve_qkv', 'recompute_k', 'recompute_v', 'recompute_q', 'ablate_l1']
labels = ['Original model', 'Ablate L1; preserve L2 Q, K, V',
          'Ablate L1; recompute L2 K', 'Ablate L1; recompute L2 V',
          'Ablate L1; recompute L2 Q', 'Ablate L1; recompute all L2 Q, K, V']
values = [100 * s['results'][c]['iwl_copy_avail']['in_context_acc'] for c in conditions]
colors = ['#4b5563', '#0072B2', '#D55E00', '#D55E00', '#009E73', '#8b5cf6']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
fig, ax = plt.subplots(figsize=(11.5, 5.5), layout='constrained')
pos = np.arange(len(conditions))
ax.barh(pos, values, color=colors, height=0.65)
ax.set_yticks(pos, labels)
ax.invert_yaxis()
ax.set_xlim(0, 110)
ax.set_xlabel('CIWL in-context accuracy (%)')
ax.axvline(50, color='#6b7280', linestyle='--', linewidth=1.2, label='Chance: 50%')
for p, v in zip(pos, values):
    ax.text(v + 0.6, p, f'{v:.2f}%', ha='left', va='center', fontsize=12)
ax.grid(axis='x', alpha=0.15)
ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=10)
fig.suptitle('L1 ablation and L2 Q/K/V composition\nSame saved 20M model; same 5,000 fixed CIWL sequences', fontsize=16)
fig.savefig(outdir / 'ciwl_ablations.png', dpi=190, bbox_inches='tight')
fig.savefig(outdir / 'ciwl_ablations.svg', bbox_inches='tight')
plt.close(fig)
print('plots_completed', flush=True)
