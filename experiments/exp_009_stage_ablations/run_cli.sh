#!/usr/bin/env bash
# Parameterize unchanged author CLI; preserve the original scientific sources.
set -euo pipefail
task_root="$HOME/research/icl-dynamics/exp_009_stage_ablations"
research_root="$HOME/research/icl-dynamics"
cd "$task_root"
mkdir -p logs outputs runs
trap 'code=$?; printf "%s\n" "$code" > exit_code; date -u +%FT%TZ > finished' EXIT
date -u +%FT%TZ >> starts
test -e started || date -u +%FT%TZ > started
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export PYTHONPATH="$research_root/exp_001_hardware_check/source:$task_root/author_reference"
task_python="$research_root/exp_001_hardware_check/venv/bin/python"
"$task_python" -m pip freeze > outputs/environment.txt
"$task_python" - <<'PY' > logs/preflight.log 2>&1
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json
import numpy as np
import h5py,jax,jax.numpy as jnp,equinox as eqx
import main_utils,opto
w=Path.cwd();root=w.parent;source=root/'exp_001_hardware_check/source'
e=json.loads((w/'expected_baseline.json').read_text());plan=json.loads((w/'plan.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=root/'exp_002_main_reproduction/runs/main/config.json'
ev=root/'exp_002_main_reproduction/runs/default_eval/eval_data.h5'
assert jax.default_backend()=='gpu' and jax.config.jax_default_matmul_precision is None
assert sha(cfg)==e['configuration_sha256'] and sha(ev)==e['evaluation_data_sha256']
for n,h in e['source_sha256'].items():assert sha(source/n)==h,n
for n,h in e['author_reference_sha256'].items():assert sha(w/'author_reference'/n)==h,n
for n,k in [('summary.json','reference_exp008_summary_sha256'),('provenance.json','reference_exp008_provenance_sha256')]:
 assert sha(w/'reference_exp008'/n)==e[k]
ref=json.loads((w/'reference_exp008/provenance.json').read_text())
assert ref['checkpoint_sha256']==e['checkpoint_sha256']
assert ref['configuration_sha256']==e['configuration_sha256'] and ref['evaluation_sha256']==e['evaluation_data_sha256']
assert ref['source_sha256']==e['source_sha256'] and ref['author_reference_sha256']==e['author_reference_sha256']
assert ref['eval_batch_size']==1024 and ref['eval_key']==[0,0]
opts=main_utils.get_opts_from_json_file(str(cfg));m0=main_utils.get_model_from_opts(opts,(512,))
key=jax.random.PRNGKey(0)
template=dict(iter=-1,model=m0,opt_state=main_utils.get_optimizer_from_opts(opts).init(eqx.filter(m0,eqx.is_array)),seeds=dict(eval_model_seed=key,train_data_seed=key,train_model_seed=key))
parser=argparse.ArgumentParser();opto.add_args_to_parser(parser)
fwd_from_train=opto.make_fn_from_opts(opts)
data={}
with h5py.File(ev,'r') as f:
 for n in ['icl','iwl_copy_avail','pure_iwl']:
  assert f[n]['examples'].shape==(5000,3,512) and f[n]['labels'].shape==(5000,3)
  y=f[n]['labels'][:]
  matches=(y[:,:2]==y[:,-1,None]).sum(1)
  assert np.all(matches==(0 if n=='pure_iwl' else 1))
  if n=='icl':assert np.all(np.isin(y,[0,1]))
  data[n]=(jnp.asarray(f[n]['examples'][0]),jnp.asarray(y[0]))
audit={};checkpoints={}
for stage in [20000000,4000000,8000000]:
 cp=root/f'exp_002_main_reproduction/runs/main_resume_00002950016/checkpoints/{stage:011d}.eqx'
 assert cp.stat().st_size==e['checkpoint_bytes'] and sha(cp)==e['checkpoint_sha256_by_stage'][str(stage)]
 ck=eqx.tree_deserialise_leaves(cp,template);model=ck['model']
 assert int(ck['iter'])==stage and int(ck['opt_state'][0].count)==stage//32
 assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(ck) if eqx.is_array(v))
 assert len(model.transformer.blocks)==2 and opts.mlp_ratio is None
 assert all(b.attn.proj.bias is None and b.drop_path2.__class__.__name__=='Zeros' for b in model.transformer.blocks)
 checkpoints[str(stage)]=dict(sha256=sha(cp),iter=stage,adam_count=stage//32)
 audit[str(stage)]={}
 used=set()
 for job in [p for p in plan if p['stage']==stage and p['evaluator']=='icl']:
  name=job['condition']
  if name in used:continue
  used.add(name);o=parser.parse_args(job['flags'])
  preserve=[q for q,attr in [('q','opto_preserve_queries'),('k','opto_preserve_keys'),('v','opto_preserve_values')] if getattr(o,attr)]
  call=opto.make_fn_from_opts(deepcopy(o),default_fn=fwd_from_train)
  # Reuse exp_006's native internal-cache capture inside one compiled graph.
  def paired_call(m,x,y,k):
   class CaptureModel:
    def __init__(self,native):
     self.native=native;self.transformer=native.transformer;self.calls=[]
    def call_with_all_aux(self,**kwargs):
     result=self.native.call_with_all_aux(**kwargs);self.calls.append(result);return result
   captured=CaptureModel(m);changed=call(captured,x,y,k)
   assert len(captured.calls)==(2 if preserve else 1)
   original=captured.calls[0] if preserve or name=='baseline' else fwd_from_train(m,x,y,k)
   return original,changed
  x,y=data['icl'];original,changed=eqx.filter_jit(paired_call)(model,x,y,key)
  blocks=changed['transformer_output']['block_outputs'];values={}
  if name=='embedding_only':
   for b in blocks:
    np.testing.assert_array_equal(b['attn_output']['v'],np.zeros_like(b['attn_output']['v']))
    np.testing.assert_array_equal(b['attn_output']['out'],np.zeros_like(b['attn_output']['out']))
    np.testing.assert_array_equal(b['out'],changed['embedding'])
   np.testing.assert_array_equal(changed['transformer_output']['pre_unembed'],jax.vmap(model.transformer.norm)(changed['embedding']))
   for n in ['icl','iwl_copy_avail','pure_iwl']:
    x,y=data[n]
    r=call(model,x,y,key)
    xx=x.at[:2].set(x[:2][::-1]);yy=y.at[:2].set(y[:2][::-1])
    r2=call(model,xx,yy,key)
    np.testing.assert_array_equal(r['out'][-1],r2['out'][-1])
   values={'both_attention_branches_zero':True,'both_layer_outputs_equal_embedding':True,'final_norm_receives_embedding':True,'query_logits_invariant_to_context_permutation_all3_evaluators':True}
  else:
   if name!='baseline':
    np.testing.assert_array_equal(blocks[0]['attn_output']['v'],np.zeros_like(blocks[0]['attn_output']['v']))
    np.testing.assert_array_equal(blocks[0]['out'],changed['embedding'])
   source_qkv=original['transformer_output']['block_outputs'][1]['attn_output']
   for q in ['q','k','v']:
    a=np.asarray(blocks[1]['attn_output'][q]);b=np.asarray(source_qkv[q])
    diff=float(np.max(np.abs(a-b)));retained=name=='baseline' or q in preserve
    if retained:np.testing.assert_allclose(a,b,rtol=2e-5,atol=3e-6)
    else:assert diff>1e-5,(stage,name,q)
    values[q]=dict(preserved=retained,max_abs_change=diff)
  audit[str(stage)][name]=values
  print('audit',stage,name,json.dumps(values),flush=True)
prov=dict(checkpoints=checkpoints,configuration_sha256=sha(cfg),evaluation_sha256=sha(ev),source_sha256=e['source_sha256'],author_reference_sha256=e['author_reference_sha256'],jax=jax.__version__,equinox=eqx.__version__,devices=[str(d) for d in jax.devices()],eval_batch_size=1024,eval_key=[0,0],precision='float32, original default matmul precision',plan=plan,no_training=True,unchanged_author_cli=True,rtol=2e-5,atol=3e-6)
(w/'outputs/provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
(w/'outputs/intervention_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('All checkpoint identities and native intervention audits passed.',flush=True)
PY
while IFS=$'\t' read -r jid stage condition evaluator flags; do
  if test -e "logs/$jid.done"; then continue; fi
  mkdir -p "runs/$jid/checkpoints"
  ln -sfn "$research_root/exp_002_main_reproduction/runs/main/config.json" "runs/$jid/config.json"
  ln -sfn "$research_root/exp_002_main_reproduction/runs/main_resume_00002950016/log.h5" "runs/$jid/log.h5"
  printf -v ckname '%011d.eqx' "$stage"
  ln -sfn "$research_root/exp_002_main_reproduction/runs/main_resume_00002950016/checkpoints/$ckname" "runs/$jid/checkpoints/$ckname"
  if [[ "$evaluator" == pure_iwl ]]; then metric=prob; else metric=in_context_prob; fi
  args=(--base_folder "$task_root/runs" --run_folder "$jid"
    --plots metric_by_color_by --color_by default --baseline_metric "$metric"
    --metric_range 0 1 --metric_curve eval_iter train_eval/loss
    --loss_curve eval_iter train_eval/loss --integral_metric_start "$stage"
    --data_mode eval --data_file "$research_root/exp_002_main_reproduction/runs/default_eval/eval_data.h5"
    --eval_subsets "$evaluator" --batch_size 1024
    --plot_range "$stage" "$stage" --num_ckpts_to_plot 1
    --only_plot_avg --save_plot_as pkl png)
  if [[ -n "$flags" ]]; then read -r -a opto_args <<< "$flags"; args+=("${opto_args[@]}"); fi
  printf '%q ' "$task_python" "$task_root/author_reference/visualize_runs.py" "${args[@]}" >> outputs/commands.txt
  printf '\n' >> outputs/commands.txt
  date -u +%FT%TZ > "logs/$jid.started"
  "$task_python" -u "$task_root/author_reference/visualize_runs.py" "${args[@]}" > "logs/$jid.log" 2>&1
  date -u +%FT%TZ > "logs/$jid.done"
  "$task_python" -u read_results.py >> logs/results.log 2>&1
done < <("$task_python" - <<'PY'
import json
for p in json.load(open('plan.json')):
 print('\t'.join([p['job_id'],str(p['stage']),p['condition'],p['evaluator'],' '.join(p['flags'])]))
PY
)
"$task_python" - <<'PY' > logs/final_check.log 2>&1
from pathlib import Path
import json,hashlib
w=Path.cwd();e=json.loads((w/'expected_baseline.json').read_text())
rows=json.loads((w/'outputs/summary.json').read_text())
assert len(rows)==len(json.loads((w/'plan.json').read_text()))==28
for row in rows:
 if row['condition']=='baseline':assert abs(row['baseline_log_difference'])<=0.0002+1e-12,row
for stage,digest in e['checkpoint_sha256_by_stage'].items():
 p=w.parent/f'exp_002_main_reproduction/runs/main_resume_00002950016/checkpoints/{int(stage):011d}.eqx'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
print('All28 author CLI evaluations complete; baseline identities and unchanged checkpoints verified.')
PY
