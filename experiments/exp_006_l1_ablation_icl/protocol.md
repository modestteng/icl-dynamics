# Supplement all six L1/L2-QKV ablations with fixed ICL evaluation

Authorized 2026-10-04: the user asks to evaluate ICL after the same six ablations, since CIWL dominance before ablation does not settle ICL behavior after intervention.

This is an ICL evaluation extension of exp_005. Keep the original exp_005 results and source snapshots. No training, new checkpoint, data resampling, new condition, or new selection threshold is involved.

## Frozen implementation and inputs

- Use exactly the validated exp_005 runner's six opto conditions and exported author notebook `make_forward_fn_ic_only`, original `opto.make_fn_from_opts`, and `main.make_batched_fn`. Baseline, preserve-QKV, recompute-K, recompute-V, recompute-Q, and fully ablated L1 conditions are unchanged.
- The implement-research-code skill remains unavailable after local discovery. Continue the existing user-authorized minimal adaptation, retaining unchanged author scientific source. Only add the previously omitted evaluator and comparison/reporting code.
- Same exp_002 final checkpoint at 20,000,000 sequences / 625,000 Adam updates; same original config, source hashes, evaluation-file hash and original float32/default precision.
- Evaluate the original `icl` group, all 5000 sequences in unchanged order. Labels must be 0/1, with the query label present exactly once among the two context labels. The ICL evaluator invalidates original fixed exemplar-label mappings and tests using the contextual mapping (paper Section2.3).
- Evaluation batch=1024 and PRNGKey(0) remain identical to exp_005 / author notebook. Primary metric is `in_context_acc`, chance=0.5. Retain per-sequence probability, contextual loss and full-space loss from the exact author function.
- Reuse frozen exp_005 summary and provenance as separately identified reference artifacts. Verify their SHA-256, checkpoint/config/evaluation/source identities, function hash, flags, batch, key and precision before comparing existing CIWL/Flip results. Do not rerun the existing three evaluation sets.
- Directly trace the author's native internal cache on ICL row0 to verify L1 V ablation and L2 preservation/recomputation, using the successful exp_005 audit method and unchanged tolerances. Independently computed original forwards are not used as source-cache identities.
- Windows RTX4070/WSL original JAX0.4.26/Equinox0.11.4 environment and persistent task. Expected computation is tens of seconds, below3 hours.

## Analysis fixed before execution

Report all six ICL accuracy means and correct counts alongside the previously measured CIWL and Flip values, regardless of direction. Compare every ICL condition to its original-model ICL baseline using paired correct/wrong changes and normal95 intervals. Retain contextual losses and probabilities so an accuracy change is not conflated with all aspects of prediction quality. These are descriptive fixed-model results; intervals reflect evaluation-sequence variation, not independent training-seed uncertainty or a familywise multiple-testing claim.

The original20M ICL log accuracy is0.5408. Record any recomputation difference instead of replacing the original log. An ablation's higher ICL accuracy is an effect of intervention on this existing model, not evidence of new training or universally improved ICL.

Paper Section4 focuses on asymptotic CIWL; notebook cells17/33 omit the independent ICL set. No explicit author explanation for that particular omission has been found. Do not infer that ICL cannot return after ablation, or claim the whole paper failed to evaluate ICL.

Save protocol/code snapshots, commands, environment, input/artifact identities, logs, tensor audit, all per-sequence metrics, summary, comparison plot and delivery hashes. Record idea -> change -> result -> keep/discard/inconclusive and update notes/results. This does not execute the distinct proposed L1-only readout experiment.
