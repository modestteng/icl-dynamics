#!/usr/bin/env bash
# Launch unchanged author CLI; no model/forward/evaluation source edits.
set -euo pipefail
task_root="$HOME/research/icl-dynamics/exp_008_l2_ablation"
research_root="$HOME/research/icl-dynamics"
cd "$task_root"
mkdir -p logs outputs runs
trap 'code=$?; printf "%s\n" "$code" > exit_code; date -u +%FT%TZ > finished' EXIT
date -u +%FT%TZ > started
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export PYTHONPATH="$research_root/exp_001_hardware_check/source:$task_root/author_reference"
task_python="$research_root/exp_001_hardware_check/venv/bin/python"
"$task_python" -m pip freeze > outputs/environment.txt
"$task_python" - <<'PY' > logs/preflight.log 2>&1
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import h5py,jax,jax.numpy as jnp,equinox as eqx
import main_utils,opto
w=Path.cwd(); root=w.parent; source=root/'exp_001_hardware_check/source'
e=json.loads((w/'expected_baseline.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ckpath=Path(e['checkpoint']);cfg=root/'exp_002_main_reproduction/runs/main/config.json'
ev=root/'exp_002_main_reproduction/runs/default_eval/eval_data.h5'
assert jax.default_backend()=='gpu'
assert jax.config.jax_default_matmul_precision is None
assert ckpath.stat().st_size==e['checkpoint_bytes']
assert sha(ckpath)==e['checkpoint_sha256']
assert sha(cfg)==e['configuration_sha256']
assert sha(ev)==e['evaluation_data_sha256']
for n,h in e['source_sha256'].items():assert sha(source/n)==h,n
for n,h in e['author_reference_sha256'].items():assert sha(w/'author_reference'/n)==h,n
opts=main_utils.get_opts_from_json_file(str(cfg));m0=main_utils.get_model_from_opts(opts,(512,))
key=jax.random.PRNGKey(0)
template=dict(iter=-1,model=m0,opt_state=main_utils.get_optimizer_from_opts(opts).init(eqx.filter(m0,eqx.is_array)),seeds=dict(eval_model_seed=key,train_data_seed=key,train_model_seed=key))
ck=eqx.tree_deserialise_leaves(ckpath,template);m=ck['model']
assert int(ck['iter'])==20000000 and int(ck['opt_state'][0].count)==625000
assert len(m.transformer.blocks)==2 and opts.mlp_ratio is None
assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(ck) if eqx.is_array(v))
assert m.transformer.blocks[1].attn.num_heads==8
assert m.transformer.blocks[1].attn.proj.bias is None
assert m.transformer.blocks[1].drop_path2.__class__.__name__=='Zeros'
p=argparse.ArgumentParser();opto.add_args_to_parser(p)
a=p.parse_args(['--opto_ablate_heads']+[f'1:{h}' for h in range(8)])
call=opto.make_fn_from_opts(a)
audit={}
with h5py.File(ev,'r') as f:
 for n in ['icl','iwl_copy_avail']:
  assert f[n]['examples'].shape==(5000,3,512) and f[n]['labels'].shape==(5000,3)
  yall=f[n]['labels'][:]
  assert np.all((yall[:,:2]==yall[:,-1,None]).sum(1)==1)
  if n=='icl':assert np.all(np.isin(yall,[0,1]))
  audit[n]=[]
  for i in [0,4999]:
   x=jnp.asarray(f[n]['examples'][i]);y=jnp.asarray(f[n]['labels'][i])
   original=m.call_with_all_aux(examples=x,labels=y,key=key)
   r=call(m,x,y,key)
   blocks=r['transformer_output']['block_outputs']
   np.testing.assert_array_equal(blocks[0]['out'],original['transformer_output']['block_outputs'][0]['out'])
   np.testing.assert_array_equal(blocks[1]['attn_output']['v'],np.zeros_like(blocks[1]['attn_output']['v']))
   np.testing.assert_array_equal(blocks[1]['attn_output']['out'],np.zeros_like(blocks[1]['attn_output']['out']))
   np.testing.assert_array_equal(blocks[1]['out'],blocks[0]['out'])
   expected_pre=jax.vmap(m.transformer.norm)(blocks[0]['out'])
   np.testing.assert_array_equal(r['transformer_output']['pre_unembed'],expected_pre)
   audit[n].append(dict(row=i,l1_unchanged=True,l2_v_zero=True,l2_attention_output_zero=True,l2_output_equals_l1_output=True,original_final_norm_receives_l1_output=True))
assert sha(ckpath)==e['checkpoint_sha256']
prov=dict(checkpoint_sha256=sha(ckpath),configuration_sha256=sha(cfg),evaluation_sha256=sha(ev),source_sha256=e['source_sha256'],author_reference_sha256=e['author_reference_sha256'],jax=jax.__version__,equinox=eqx.__version__,devices=[str(d) for d in jax.devices()],eval_batch_size=1024,eval_key=[0,0],precision='float32, original default matmul precision',conditions={'baseline':[],'ablate_l2':['--opto_ablate_heads']+[f'1:{h}' for h in range(8)]},no_training=True,unchanged_author_cli=True,full_layer_bypass_equivalence='no MLP; output projection bias None; all L2 V zero => L2 branch zero => L2 out=L1 out',audited_evaluators=['icl','iwl_copy_avail'])
(w/'outputs/provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
(w/'outputs/intervention_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('Preflight identities and direct-L1-readout equivalence audit passed.',flush=True)
PY
for condition in baseline ablate_l2; do
  for evaluator in icl iwl_copy_avail; do
    run_name="${condition}_${evaluator}"
    mkdir -p "runs/$run_name/checkpoints"
    ln -s "$research_root/exp_002_main_reproduction/runs/main/config.json" "runs/$run_name/config.json"
    ln -s "$research_root/exp_002_main_reproduction/runs/main_resume_00002950016/log.h5" "runs/$run_name/log.h5"
    ln -s "$research_root/exp_002_main_reproduction/runs/main_resume_00002950016/checkpoints/00020000000.eqx" "runs/$run_name/checkpoints/00020000000.eqx"
    args=(--base_folder "$task_root/runs" --run_folder "$run_name"
      --plots metric_by_color_by --color_by default --baseline_metric in_context_prob
      --metric_range 0 1 --metric_curve eval_iter train_eval/loss
      --loss_curve eval_iter train_eval/loss --integral_metric_start 20000000
      --data_mode eval --data_file "$research_root/exp_002_main_reproduction/runs/default_eval/eval_data.h5"
      --eval_subsets "$evaluator" --batch_size 1024
      --plot_range 20000000 20000000 --num_ckpts_to_plot 1
      --only_plot_avg --save_plot_as pkl png)
    if [[ "$condition" == ablate_l2 ]]; then
      args+=(--opto_ablate_heads 1:0 1:1 1:2 1:3 1:4 1:5 1:6 1:7)
    fi
    printf '%q ' "$task_python" "$task_root/author_reference/visualize_runs.py" "${args[@]}" >> outputs/commands.txt
    printf '\n' >> outputs/commands.txt
    date -u +%FT%TZ > "logs/$run_name.started"
    "$task_python" -u "$task_root/author_reference/visualize_runs.py" "${args[@]}" > "logs/$run_name.log" 2>&1
    date -u +%FT%TZ > "logs/$run_name.finished"
  done
done
# Read stored CLI metrics; no additional model evaluations.
"$task_python" - <<'PY' > logs/results.log 2>&1
from pathlib import Path
import hashlib,json,pickle,csv
import numpy as np
import h5py
w=Path.cwd();rows=[];stored={};e=json.loads((w/'expected_baseline.json').read_text())
with h5py.File(w/'outputs/per_sequence_metrics.h5','w') as h:
 for condition in ['baseline','ablate_l2']:
  stored[condition]={}
  for evaluator in ['icl','iwl_copy_avail']:
   files=list((w/'runs'/f'{condition}_{evaluator}'/'plots').glob('*.pkl'))
   assert len(files)==1
   with files[0].open('rb') as f:info=pickle.load(f)['info']
   assert info['iters']==[20000000]
   arrays={k:np.asarray(info[k][0]) for k in ['in_context_acc','in_context_prob','in_context_loss','loss','acc','prob']}
   assert all(a.shape==(5000,) and np.isfinite(a).all() for a in arrays.values())
   stored[condition][evaluator]=arrays
   g=h.create_group(f'{condition}/{evaluator}')
   for k,a in arrays.items():g.create_dataset(k,data=a,compression='gzip')
   row=dict(condition=condition,evaluator=evaluator,count=5000,correct_count=int(arrays['in_context_acc'].sum()),**{k:float(a.mean(dtype=np.float64)) for k,a in arrays.items()})
   rows.append(row);print(json.dumps(row),flush=True)
   if condition=='baseline':assert abs(row['in_context_acc']-e['baseline_logged_accuracy'][evaluator])<=0.0002+1e-12
 for evaluator in ['icl','iwl_copy_avail']:
  before=stored['baseline'][evaluator]['in_context_acc'].astype(np.int8)
  after=stored['ablate_l2'][evaluator]['in_context_acc'].astype(np.int8)
  g=h.create_group(f'paired/{evaluator}')
  g.attrs['correct_to_wrong']=int(((before==1)&(after==0)).sum())
  g.attrs['wrong_to_correct']=int(((before==0)&(after==1)).sum())
  g.attrs['accuracy_change']=float((after-before).mean())
with (w/'outputs/metrics.tsv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');wr.writeheader();wr.writerows(rows)
(w/'outputs/summary.json').write_text(json.dumps(rows,indent=2)+'\n')
ck=Path(e['checkpoint'])
assert hashlib.sha256(ck.read_bytes()).hexdigest()==e['checkpoint_sha256']
print('All four fixed-evaluator results retained; checkpoint unchanged.',flush=True)
PY
