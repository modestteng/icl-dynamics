import argparse,json,time
from pathlib import Path
import h5py,numpy as np
root=Path.home()/'research/icl-dynamics/exp_002_main_reproduction'
parser=argparse.ArgumentParser()
parser.add_argument('--run',default='main')
args=parser.parse_args()
run=root/'runs'/args.run
result={}
config=json.loads((run/'config.json').read_text())
closed=(root/'finalization_20m.finished').exists() and (root/'final_checkpoint_audit.json').exists()
with h5py.File(run/'log.h5','r',locking=False) as f:
 # Observe only an older immutable prefix, without locking out the writer.
 # Skip the newest row because append-only HDF5 output is not SWMR.
 observed_x=np.asarray(f['eval_iter'])
 n=min([len(observed_x)]+[f[k+'/'+m].shape[0] for k,m in
       [('icl','in_context_acc'),('iwl_copy_avail','in_context_acc'),
        ('flip_icl','in_context_acc'),('pure_iwl','acc'),('train_eval','in_context_acc')]])-(0 if closed else 1)
 assert n>0, 'No older committed evaluator row available yet; retry later'
 x=observed_x[:n]
 result['snapshot_policy']=('closed log; all committed rows' if closed else 'conservative older prefix; newest evaluation row excluded; no reader file lock')
 result['evaluation_points']=len(x)
 result['last_evaluated_sequences']=int(x[-1])
 start=int(Path(config['load_from_ckpt']).stem) if config.get('load_from_ckpt') else 0
 train_rows=(int(x[-1])-start)//config['train_bs']
 assert 0<train_rows<=min(len(f['train_iter']),len(f['train_loss']))
 result['last_logged_training_sequences']=int(f['train_iter'][train_rows-1])
 result['last_logged_loss']=float(f['train_loss'][train_rows-1])
 result['metrics']={}
 for k,metric in [('icl','in_context_acc'),('iwl_copy_avail','in_context_acc'),('flip_icl','in_context_acc'),('pure_iwl','acc'),('train_eval','in_context_acc')]:
  values=np.asarray(f[k+'/'+metric][:n])
  assert np.isfinite(values).all()
  y=values.mean(axis=1)
  result['metrics'][k]={'metric':metric,'initial':float(y[0]),'latest':float(y[-1]),'maximum':float(y.max()),'maximum_at_sequences':int(x[y.argmax()])}
ckpts=sorted((run/'checkpoints').glob('*.eqx'))
audit_path=root/'resume_audit.json'
if audit_path.exists():
 size=json.loads(audit_path.read_text())['checkpoint_bytes']
 ckpts=[path for path in ckpts if path.stat().st_size==size]
result['checkpoint_count']=len(ckpts)
result['last_checkpoint']=ckpts[-1].name if ckpts else None
result['run_segment']=args.run
result['total_sequences']=config['train_iters']
if closed:
 result['original_author_target_sequences']=config['train_iters']
 result['total_sequences']=json.loads((root/'final_checkpoint_audit.json').read_text())['iter']
 result['experiment_status']='completed_user_approved_integer_endpoint'
result['feature_manifest']=json.loads((root/'features/feature_manifest.json').read_text())
with h5py.File(root/'features/omniglot_features_all.h5','r') as f:
 result['encoded_vectors']=int(f.attrs['completed_vectors'])
 result['encoding_seconds']=float(f.attrs['encoding_seconds_this_invocation'])
print(json.dumps(result))
