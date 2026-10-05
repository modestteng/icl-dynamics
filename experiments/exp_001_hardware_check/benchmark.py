"""Time the author's unchanged main-model computation, not scientific results.

Random placeholder feature tensors have the full paper dimensions. They are
ONLY for estimating memory and compute; no accuracy/ICL conclusion is reported.
The original samplers, forward, CE, Adam and train_step are called directly.
"""
import argparse
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--steps", type=int, default=200)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import main
import main_utils
import opto
import samplers

started = perf_counter()
assert jax.default_backend() == "gpu", jax.devices()
opts = main_utils.create_parser().parse_args([
    "--data_type", "file", "--pos_embedding_type", "ape",
    "--depth", "2", "--d_model", "64", "--num_heads", "8",
    "--class_split", "12800", "0", "184",
    "--exemplar_split", "20", "0", "0",
    "--init_seed", "5", "--train_seed", "2",
    "--mixing_coeffs", "1", "--pt_burstiness", "1",
    "--train_context_len", "2", "--fs_relabel", "0",
    "--lr", "0.00001", "--optimizer", "adam",
])
opts.model_output_classes = 12800
config = dict(vars(opts))
model = main_utils.get_model_from_opts(opts, input_shape=(512,))
optimizer = main_utils.get_optimizer_from_opts(opts)
state = optimizer.init(eqx.filter(model, eqx.is_array))
parameter_count = sum(a.size for a in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
data = jnp.asarray(np.random.default_rng(0).standard_normal((12984, 20, 512), dtype=np.float32))
data.block_until_ready()
classes = jnp.arange(12800)
class_distr = jnp.ones(12800, dtype=jnp.float32) / 12800
substrate = partial(samplers.get_constant_burst_seq_idxs,
                    classes=classes, class_distr=class_distr,
                    num_seqs=32, context_len=2, burstiness=1,
                    distractor=1, no_support=0, unique_rest=0)
class_sampler = partial(samplers.get_mixed_seq_idxs,
                        mix_probabilities=jnp.array([1.]), mix_substrate_fns=[substrate])
exemplar_sampler = partial(samplers.get_exemplar_inds,
                           allowed_inds=jnp.arange(20), match_query_and_distractors=False)
sampler = jax.jit(partial(samplers.make_data_sampler(
    class_sampler, exemplar_sampler, fs_relabel=None,
    noise_scale=0., assign_query_label_random=False), data=data))
key = jax.random.PRNGKey(2)
first_step = True

def step():
    global key, model, state, first_step
    key, data_key, model_key = jax.random.split(key, 3)
    if first_step:
        print('phase first_sampler_enter', flush=True)
    batch = sampler(data_key)
    if first_step:
        jax.block_until_ready(batch)
        print('phase first_sampler_complete; train_step_enter', flush=True)
    metrics, model, state = main.train_step(
        model=model, fwd_fn=opto.default_model_fwd_fn, optimizer=optimizer,
        opt_state=state, microbs=32, weight_decay=0.,
        x=batch['examples'], y=batch['labels'], key=model_key)
    # Synchronize leaves individually. The 0.4.26 batched synchronization path
    # stalled on this WSL host; it is probe-only and absent from author training.
    for leaf in jax.tree_util.tree_leaves((metrics, model, state)):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()
    if first_step:
        print('phase first_train_step_complete', flush=True)
        first_step = False

print('GPU', jax.devices(), 'parameters', parameter_count, flush=True)
warm_start = perf_counter()
setup_seconds = warm_start - started
for _ in range(3):
    step()
warm_seconds = perf_counter() - warm_start
print('warmup_seconds', warm_seconds, flush=True)
times = []
for index in range(args.steps):
    start = perf_counter()
    step()
    times.append(perf_counter() - start)
    if (index + 1) % 50 == 0:
        print('timed_steps', index + 1, 'mean_seconds', float(np.mean(times)), flush=True)

# Main sweep evaluates 5 sets x 5000 sequences, using eval_bs=1000.
# Time the real eval_step at that batch size, then extrapolate to all sets.
eval_batch = sampler(jax.random.PRNGKey(11))
eval_x = jnp.tile(eval_batch['examples'], (32, 1, 1))[:1000]
eval_y = jnp.tile(eval_batch['labels'], (32, 1))[:1000]
eval_x.block_until_ready()
eval_key = jax.random.PRNGKey(1)
eval_start = perf_counter()
out = main.eval_step(model=model, fwd_fn=opto.default_model_fwd_fn,
                     x=eval_x, y=eval_y, key=eval_key)
jax.block_until_ready(out)
eval_compile_seconds = perf_counter() - eval_start
eval_times = []
for _ in range(3):
    start = perf_counter()
    out = main.eval_step(model=model, fwd_fn=opto.default_model_fwd_fn,
                         x=eval_x, y=eval_y, key=eval_key)
    jax.block_until_ready(out)
    eval_times.append(perf_counter() - start)

ckpt = dict(iter=0, seeds=dict(eval_model_seed=eval_key,
                              train_data_seed=key, train_model_seed=key),
            opt_state=state, model=model)
args.out.parent.mkdir(parents=True, exist_ok=True)
ckpt_path = args.out.with_suffix('.timing.eqx')
start = perf_counter()
eqx.tree_serialise_leaves(ckpt_path, ckpt)
checkpoint_seconds = perf_counter() - start
report = dict(
    purpose='resource estimation only; random placeholder features; NOT a reproduction result',
    config=config, devices=[str(d) for d in jax.devices()], parameters=parameter_count,
    feature_shape=list(data.shape), feature_bytes=data.size * data.dtype.itemsize,
    setup_seconds=setup_seconds, warmup_seconds=warm_seconds, timed_steps=args.steps,
    step_mean_seconds=float(np.mean(times)), step_median_seconds=float(np.median(times)),
    step_p90_seconds=float(np.percentile(times, 90)),
    eval_batch_size=1000, eval_compile_seconds=eval_compile_seconds,
    eval_mean_seconds=float(np.mean(eval_times)),
    checkpoint_seconds=checkpoint_seconds, checkpoint_bytes=ckpt_path.stat().st_size,
    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    total_seconds=perf_counter() - started,
    versions={n:importlib.metadata.version(n) for n in
              ['jax','jaxlib','equinox','optax','numpy','scipy','h5py','jaxtyping']})
args.out.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
