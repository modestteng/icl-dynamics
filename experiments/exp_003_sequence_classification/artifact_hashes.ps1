$ErrorActionPreference='Stop'
$root='D:\research\icl-dynamics\exp_003_sequence_classification'
$names=@('validation.json','classification_summary.json','llm_review_samples.json','score_provenance.json','progress.json','final_audit.json','heldout_metrics.npz','final_audit.log','review_decision.json','full_review_audit.json')
$results=foreach($n in $names){ $p=Join-Path $root $n; @{name=$n;sha256=(Get-FileHash -Algorithm SHA256 $p).Hash.ToLower();bytes=(Get-Item $p).Length} }
$results|ConvertTo-Json -Depth 4
