# Figure3b: reuse the 20M second half with five saved first halves

Authorized2026-10-04. User asks to reproduce the identified Figure3b result using the existing20M model, then explicitly narrows the work to enough representative cases to test its core conclusion.

## Frozen scope

- Five first-half checkpoints selected before graft evaluation:0,2M,4.4M,8M,20M. The4.4M point is the previously recorded native-model ICL peak, not selected using graft results. Preserve every outcome; do not add checkpoints to search for agreement.
- Fixed second half from the existing20M checkpoint: L2, final normalization, unembedding. The variable first half includes input embeddings and L1, exactly matching the paper's partition.
- Reuse unchanged author `opto.make_fn_from_opts`, with `opto_graft_out_model_ckpt=20M`, original config, and `opto_graft_out_model_from_layer=0`. This injects the current first half's full output at the boundary and lets the fixed second half recompute its Q/K/V. All head-ablation and Q/K/V/attention-pattern-preservation flags are absent/false. No training or deletion of a layer occurs.
- Use the original author notebook's exported metric forward and batching function. Metric computation is the same in-context accuracy used for Figure3b. Keep defaultfloat32 precision, batch1024 and PRNGKey(0).
- Original fixed train_eval,icl,iwl_copy_avail(CIWL),flip_icl, each5000 sequences. No sampling or relabeling. Five grafted evaluations total100000 sequences. Original201-point baseline curve is a frozen existing reference, not newly computed data.
- One native20M control across the same four evaluators checks identity at the self-graft endpoint:20000 additional evaluations. Record any numerical differences, including differing individual predictions, without altering precision or replacing original logs.
- Scientific implementation is reused from the authoritative local author sources. implement-research-code remains unavailable; retain the existing explicit user permission for minimal independent adapters, with author source unchanged.
- Windows RTX4070/WSL existing JAX0.4.26/Equinox0.11.4 environment, persistent task. Based on earlier full-size evaluations, expected computation is around a minute, clearly below3hours; no separate benchmark.

## Identity and intervention verification

Verify source/config/evaluation/checkpoint/reference/function hashes, all five checkpoint iter/Adam count and finite arrays. Retrieve the donor model closed over by the author's graft function and require its parameter arrays to be bitwise identical to the separately restored20M model. Check donor identity again after evaluation.

On fixed ICL row0 at each stage, trace the source call actually used by the author graft function, and require exact equality of the source L1 output, injected boundary output and donor L2 input. Save the recomputed L2 Q/K/V tensors. These checks establish the changed boundary input and fixed weight ownership; they do not compare separately compiled forwards as if their intermediate tensors were the same cache source.

No hard equality-to-paper numerical threshold is invented. Report original vs grafted accuracy at all five stages, native20M vs self-graft consistency, whether ICL rises above chance around the recorded peak then falls towards the endpoint, and whether CIWL rises. Also report deviations from native baseline, including early-phase differences. Five measurements can test core direction but do not reproduce the paper's dense full-time curve or locate a new exact peak/onset.

Keep per-sequence accuracy/probability/contextual loss/full-space loss, original curve, all flags/versions/identities, commands, logs, code snapshots, tensor audit, plot and delivery hashes. This reproduces Section5.1 Figure3b; it does not run the Section5.2 Figure4a training experiment.

## Startup audit helper correction

Attempt001 exited before evaluation when inspection of all author-function closure cells encountered an empty cell from an inactive optional branch. Retain its code and full log. Read only the named `graft_out_model` closure cell instead. The metric wrapper reuses this same verified graft callable as its default forward, avoiding another redundant donor load. Scientific graft flags, model/data/precision, all stage selections and metrics remain unchanged.
