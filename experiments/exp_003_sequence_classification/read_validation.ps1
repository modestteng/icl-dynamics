$ErrorActionPreference='Stop'
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'set -eu; cd ~/research/icl-dynamics/exp_003_sequence_classification; cp pilot/{classification_summary.json,validation.json,llm_review_samples.json,score_provenance.json,progress.json} /mnt/d/research/icl-dynamics/exp_003_sequence_classification/; cat pilot/validation.json; ls -lh /mnt/d/research/icl-dynamics/exp_003_sequence_classification/pilot_results.tar.gz'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
