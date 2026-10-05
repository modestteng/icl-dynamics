experiment_root="$HOME/research/icl-dynamics/exp_001_hardware_check"
for name in bootstrap.log install.log benchmark.log; do
 if [ -f "$experiment_root/$name" ]; then echo "$name"; tail -n 6 "$experiment_root/$name"; fi
done
pgrep -af "pip install|benchmark.py" || true
