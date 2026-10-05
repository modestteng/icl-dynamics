"""Scientific PNG/SVG plots of real checkpoint attention; no smoothing."""
from pathlib import Path
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

work=Path(__file__).resolve().parent
out=work/'outputs'
s=json.loads((out/'summary.json').read_text())
times=np.array(s['stages'])/1e6
tokens=['x1','y1','x2','y2','xq']
snapshots=[0,2000000,4400000,8000000,20000000]
blue='#0072B2';orange='#D55E00'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})

def save(fig,name):
    fig.savefig(out/f'{name}.png',dpi=190,bbox_inches='tight')
    fig.savefig(out/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

def draw(ax,m,title,numbers=True):
    mask=np.triu(np.ones((5,5),bool),1)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#f2f2f2')
    image=ax.imshow(np.ma.array(m,mask=mask),vmin=0,vmax=1,cmap=cmap)
    ax.set_xticks(range(5),tokens);ax.set_yticks(range(5),tokens)
    ax.set_title(title,fontsize=10)
    for r,c in [(1,0),(3,2)]:ax.add_patch(Rectangle((c-.48,r-.48),.96,.96,fill=False,lw=2,edgecolor='#38bdf8'))
    for r,c in [(1,1),(3,3)]:ax.add_patch(Rectangle((c-.48,r-.48),.96,.96,fill=False,lw=2,edgecolor='#ff6b35'))
    if numbers:
        for r in range(5):
            for c in range(r+1):ax.text(c,r,f'{m[r,c]:.2f}',ha='center',va='center',fontsize=8,color='black' if m[r,c]>.65 else 'white')
    return image

with h5py.File(out/'attention.h5','r') as f:
    for name in ['train_eval','icl']:
        fig,axes=plt.subplots(8,5,figsize=(17,25),layout='constrained')
        for h in range(8):
            for j,t in enumerate(snapshots):
                r=s['results'][str(t)][name]
                im=draw(axes[h,j],np.asarray(f[f'{t:011d}/{name}/mean_attention'])[0,h],
                    f'H{h} | {t/1e6:g}M\nP={r["label_prev"][h]:.2f}  S={r["label_self"][h]:.2f}')
        fig.colorbar(im,ax=axes,shrink=.4,label='Attention weight (0 to 1)')
        fig.suptitle(f'Layer 1: all 8 heads on {name}\nRows attend to columns. Blue boxes: previous sample; orange boxes: self.\nBalanced averages of the same 5,000 fixed sequences; upper-right cells are causally masked.',fontsize=14)
        save(fig,f'l1_all_heads_matrices_{name}')
    for h in range(8):
        fig,axes=plt.subplots(1,5,figsize=(17,4.3),layout='constrained')
        for j,t in enumerate(snapshots):
            r=s['results'][str(t)]['train_eval']
            im=draw(axes[j],np.asarray(f[f'{t:011d}/train_eval/mean_attention'])[0,h],
                f'{t/1e6:g}M sequences\nP={r["label_prev"][h]:.3f}  S={r["label_self"][h]:.3f}')
        fig.colorbar(im,ax=axes,shrink=.65,label='Weight')
        fig.suptitle(f'Layer 1 Head {h} (zero-based) | same 5,000 train_eval sequences\nRows = attending position; columns = attended position. Blue = previous sample, orange = self.',fontsize=13)
        save(fig,f'l1_head{h}_matrices')
    fig,axes=plt.subplots(1,5,figsize=(17,4.3),layout='constrained')
    for j,t in enumerate(snapshots):
        im=draw(axes[j],np.asarray(f[f'{t:011d}/train_eval/sample0_attention'])[0,1],f'{t/1e6:g}M sequences')
    fig.colorbar(im,ax=axes,shrink=.65,label='Weight')
    fig.suptitle('Layer 1 Head 1 | one actual fixed sequence: train_eval row 0\nSingle-example matrices, not averages; this row was selected before inspecting patterns.',fontsize=13)
    save(fig,'l1_head1_actual_sequence0')

for name in ['train_eval','icl','iwl_copy_avail','flip_icl']:
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True,layout='constrained')
    for h,ax in enumerate(axes.flat):
        p=np.array([s['results'][str(t)][name]['label_prev'][h] for t in s['stages']])
        se_p=np.array([s['results'][str(t)][name]['label_prev_se'][h] for t in s['stages']])
        q=np.array([s['results'][str(t)][name]['label_self'][h] for t in s['stages']])
        se_q=np.array([s['results'][str(t)][name]['label_self_se'][h] for t in s['stages']])
        ax.plot(times,p,'o-',color=blue,label='Previous sample')
        ax.plot(times,q,'s-',color=orange,label='Self')
        ax.fill_between(times,p-1.96*se_p,p+1.96*se_p,color=blue,alpha=.12)
        ax.fill_between(times,q-1.96*se_q,q+1.96*se_q,color=orange,alpha=.12)
        ax.axvline(4.4,color='gray',ls=':',lw=.9)
        ax.set_title(f'Head {h}');ax.set_ylim(0,1);ax.set_xlim(0,20);ax.grid(alpha=.18)
        ax.set_xlabel('Training sequences seen (million)');ax.set_ylabel('Mean attention from label positions')
    axes[0,0].legend(fontsize=9)
    fig.suptitle(f'Layer 1 on {name} | all heads, measured checkpoints only\nDotted line: 4.4M ICL peak. Bands: evaluation-sample normal 95% intervals, conditional on each fixed model.',fontsize=13)
    save(fig,f'l1_prev_self_by_head_{name}')

fig,axes=plt.subplots(2,4,figsize=(17,7),sharex=True,sharey=True,layout='constrained')
colors=plt.get_cmap('Dark2')(np.arange(8))
for j,name in enumerate(s['evaluators']):
    for h in range(8):
        axes[0,j].plot(times,[s['results'][str(t)][name]['label_prev'][h] for t in s['stages']],'-o',ms=3,color=colors[h],label=str(h))
        axes[1,j].plot(times,[s['results'][str(t)][name]['label_self'][h] for t in s['stages']],'-o',ms=3,color=colors[h],label=str(h))
    axes[0,j].set_title(name);axes[1,j].set_xlabel('Sequences seen (million)')
    for ax in axes[:,j]:ax.set_ylim(0,1);ax.set_xlim(0,20);ax.grid(alpha=.15)
axes[0,0].set_ylabel('Label to previous sample');axes[1,0].set_ylabel('Label to self')
axes[1,-1].legend(title='Head index',ncol=4,fontsize=8)
fig.suptitle('Figure 14 measurement reproduced on our saved baseline\nEqual-weight correct-label-position groups; average of the two label-token positions.',fontsize=13)
save(fig,'l1_paper14_style')
print('plots_completed',flush=True)
