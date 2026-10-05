$ErrorActionPreference='Stop'
& 'C:\Program Files\WSL\wsl.exe' -d Ubuntu-22.04 --exec /bin/bash -lc 'cd ~/research/icl-dynamics/exp_003_sequence_classification; cat logs/final_audit.log; test ! -f pilot/final_audit.json || cat pilot/final_audit.json'
