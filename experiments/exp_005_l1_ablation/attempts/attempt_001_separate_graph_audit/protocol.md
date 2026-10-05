# L1 output and Q/K/V composition ablations at the saved 20M checkpoint

User-authorized on 2026-10-04. Reproduce the experiments discussed in Sections 4.1 and 4.2, using our existing exp_002 model trained on 20,000,000 sequences. This is inference with interventions, with no optimizer steps.

## Author implementation

Reuse unchanged `opto.make_fn_from_opts`, `main.make_batched_fn`, and the exact `make_forward_fn_ic_only` function exported from cell 13 of `coopetition_paper_plots.ipynb`. Conditions reproduce cells 15, 16, and 32. The no-preservation L1 ablation is the control described in Section 4.2. Author code uses zero-based layer index 0 for paper L1 and index 1 for paper L2.

The requested implement-research-code skill remains unavailable after checking local skill/plugin locations and callable skill discovery. Existing session authorization permits an independent minimal adapter when that skill is absent (recorded in notes.md, 2026-10-02). No author computational source is edited. Notebook cells are retained with their file hash and original text.

## Frozen model, evaluation, and precision

- Final exp_002 checkpoint `00020000000.eqx`, iter=20000000, Adam count=625000. Verify SHA-256, size, full deserialization, and finite arrays before any evaluation.
- Original exp_002 config and fixed `default_eval/eval_data.h5`, with identities verified against the final baseline audit. No new sequences or labels are generated.
- Author notebook evaluation subsets: `train_eval`, `iwl_copy_avail` (CIWL), and `flip_icl`, all original 5000 sequences in unchanged order. Primary result is CIWL `in_context_acc`, chance=0.5. Retain per-sequence `in_context_acc`, `in_context_prob`, `in_context_loss`, and full-space `loss` from the original function.
- Original float32/default matmul precision. Evaluation batch_size=1024 and PRNGKey(0) match author notebook cell 12/17/33; this does not change the original training batch size of 32 or the saved model. No dropout setting is modified.
- Windows RTX 4070 / original exp_001 WSL JAX 0.4.26 and Equinox 0.11.4 environment. Expected runtime is minutes, clearly below three hours.

## Six conditions

| ID | Ablate all L1 attention heads | Preserve in L2 |
|---|---|---|
| baseline | no | normal forward |
| preserve_qkv | yes | Q, K, V |
| recompute_k | yes | Q, V; K recomputed from ablated input |
| recompute_v | yes | Q, K; V recomputed from ablated input |
| recompute_q | yes | K, V; Q recomputed from ablated input |
| ablate_l1 | yes | none; Q, K, V recomputed |

Use author opto flags directly. They preserve cached quantities before overwriting ablated L1 V tensors, exactly as in the notebook. Ablation sets the L1 attention heads' V to zero; it does not physically remove a Transformer block, its residual/normalization, or its projection bias. In the selective cases, L2 K/V/Q are recomputed rather than arbitrarily zeroed.

## Verification and reporting

Before full evaluation, verify interventions on one fixed CIWL sequence: L1 V is zero, specified L2 Q/K/V match original quantities within fixed float32 tolerance (rtol=2e-5, atol=3e-6), and recomputed quantities change. Preserve actual measured discrepancies and quantities' maximum changes in the audit. Verify full result shapes and finite metrics. Compare baseline mean accuracy to original logged values without rewriting either record.

Save code/protocol/source/checkpoint/evaluation/environment identities, opto flags, command, logs, per-sequence metrics, correct counts, accuracy means, and paired accuracy differences vs baseline. Normal 95% intervals for paired differences describe evaluation-sequence variation conditional on this fixed model, not independent training-seed uncertainty.

Paper reference CIWL percentages: baseline 98.7; preserve QKV 98.5; recompute K 57.5; recompute V 59.3; recompute Q 95.4. Fully ablated L1 is described as a large drop without a numerical value in the section text. Paper analysis uses its 64M checkpoint; ours is the user-requested 20M checkpoint, with our previously approved feature ordering. Numerical equality to paper values is not a completion gate.

Record `idea -> change -> result -> keep/discard/inconclusive`. Keep all outcomes. This replication does not include deleting L2 or training a new one-layer model.
