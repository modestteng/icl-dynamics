"""Read-only structural and confound audit; never turns a proxy into causal truth."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--pilot',type=Path,required=True)
a=p.parse_args()
with h5py.File(a.pilot/'pilot_indices.h5','r') as f:
    cls=np.asarray(f['class_idx']); ex=np.asarray(f['exemplar_idx']); exact=np.asarray(f['exact_support_example'])
result={'scope':'proxy-score audit; no per-sample causal ground truth','checkpoints':{}}
for path in sorted(a.pilot.glob('scores_*.h5')):
    with h5py.File(path,'r',locking=False) as f:
        done=int(f.attrs['completed_sequences'])
        n=done if done==1000000 else max(0,done-8192)
        if n==0:continue
        s1,s2,label=[np.asarray(f[k][:n]) for k in ['icl_score','ciwl_score','label']]
        assert np.isfinite(s1).all() and np.isfinite(s2).all()
        diff=(s1-s2)/(np.abs(s1)+np.abs(s2)+1e-12)
        expected=np.zeros(n,np.int8)
        den=np.abs(s1)+np.abs(s2)
        expected[(diff>.1)&(s1>0)&(den>1e-10)]=1
        expected[(diff<-.1)&(s2>0)&(den>1e-10)]=2
        assert np.array_equal(label,expected)
        assert np.allclose(np.asarray(f['relative_icl'][:n]),.5+.5*diff)
        bygroup={}
        for k in [0,1,2]:
            mask=label==k
            q=cls[:n,2][mask]
            bygroup[str(k)]={'count':int(mask.sum()),
                'exact_match_fraction':float(exact[:n][mask].mean()) if mask.any() else None,
                'query_label_0_or_1_fraction':float((q<2).mean()) if len(q) else None,
                'query_label_in_eval_1600_fraction':float((q<1600).mean()) if len(q) else None,
                'both_scores_positive_fraction':float(((s1[mask]>0)&(s2[mask]>0)).mean()) if mask.any() else None}
        result['checkpoints'][path.stem]={'audited_prefix':n,'labels_match_frozen_rule':True,
            'relative_score_is_not_probability':True,'bygroup':bygroup,
            'relative_score_at_0_or_1_count':int(((np.asarray(f['relative_icl'][:n])<=1e-6)|(np.asarray(f['relative_icl'][:n])>=1-1e-6)).sum()),
            'top_contrast_tie_count':{'icl':int(((diff>=1-1e-6)&(label==1)).sum()),'ciwl':int(((diff<=-1+1e-6)&(label==2)).sum())},
            'mean_scores_exact_match':[float(s1[exact[:n]].mean()),float(s2[exact[:n]].mean())],
            'mean_scores_nonidentical':[float(s1[~exact[:n]].mean()),float(s2[~exact[:n]].mean())]}
if (a.pilot/'validation_ids.npz').exists():
    with np.load(a.pilot/'validation_ids.npz') as ids:
        result['validation_selection']={}
        for name in ids.files:
            x=ids[name];q=cls[x,2]
            result['validation_selection'][name]={'count':len(x),'unique_ids':len(np.unique(x)),
                'query_label_0_or_1_fraction':float((q<2).mean()),
                'query_label_in_eval_1600_fraction':float((q<1600).mean()),
                'unique_query_classes':len(np.unique(q)),'exact_match_fraction':float(exact[x].mean())}
        result['within_seed_group_overlap']={}
        for seed in [103,104]:
            result['within_seed_group_overlap'][str(seed)]={
                f'{x}-{y}':len(np.intersect1d(ids[f'{seed}_{x}'],ids[f'{seed}_{y}']))
                for x,y in [('icl','ciwl'),('icl','random'),('ciwl','random')]}
print(json.dumps(result,indent=2))
