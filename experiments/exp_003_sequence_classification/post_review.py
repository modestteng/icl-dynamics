"""Publish the completed pilot review without replacing its frozen loss gate.

The user asks for ability formation. A passed loss gate with no stable gain on
the paper's accuracy metric does not establish that broader claim.
"""
from pathlib import Path
import json

pilot=Path.home()/'research/icl-dynamics/exp_003_sequence_classification/pilot'
v=json.loads((pilot/'validation.json').read_text())
audit=json.loads((pilot/'final_audit.json').read_text())
assert v['status']=='completed' and audit['status']=='passed_artifact_and_metric_audit'
rows=[]
for seed in v['selection_seeds']:
    groups={r['group']:r for r in v['runs'] if r['seed']==seed}
    for name,target in [('icl','icl'),('ciwl','iwl_copy_avail')]:
        s=groups[name]['after'][target];r=groups['random']['after'][target]
        rows.append(dict(seed=seed,group=name,target=target,
            loss_advantage_over_random=r['loss']-s['loss'],
            in_context_accuracy=s['in_context_acc'],random_in_context_accuracy=r['in_context_acc'],
            accuracy_advantage_percentage_points=100*(s['in_context_acc']-r['in_context_acc']),
            accuracy_change_from_anchor_percentage_points=100*(s['in_context_acc']-v['initial'][target]['in_context_acc'])))
recommendation=all(r['accuracy_advantage_percentage_points']>0 and r['accuracy_change_from_anchor_percentage_points']>0 for r in rows)
out=dict(status='completed_pilot_review',original_loss_gate_passed=v['scale_gate_passed'],
    accuracy_effects=rows,ability_growth_consistently_verified=recommendation,allow_10m=recommendation,
    frozen_validation_json_unchanged=True,review_is_posthoc=True,
    scope='auxiliary review in the current conversation plus deterministic artifact/statistical checks, not an independent causal labeling oracle',
    findings=[
        'Local full-class CE scores lower target CE in both selection seeds, but target in_context_acc does not consistently exceed random or the anchor.',
        'Reported stage-conflict count 427409 prevents interpreting labels as permanent properties of these sequences.',
        'Relative scores saturate at0/1 for opposite-sign gradients; 100% is not a calibrated probability or a strength ranking.',
        'Both screened branches degrade train_eval and Flip relative to random; adverse effects must be retained.',
        'Keep per-checkpoint continuous scores and reject labels; do not scale or claim formation labels based solely on the old loss gate.'
    ],
    script_feedback=[
        'Use this review allow_10m decision in downstream launches, not validation.scale_gate_passed alone.',
        'Future scoring revisions should examine alignment with the two in-context labels and actual Adam updates, in an independent protocol with new holdout evaluation.',
        'Do not repair a failed ability claim merely by changing the threshold on the same holdout examples.'
    ])
(pilot/'review_decision.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
