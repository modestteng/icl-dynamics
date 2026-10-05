#!/usr/bin/env bash
set -eu
cd "$HOME/research/icl-dynamics/exp_002_main_reproduction"
root="$PWD"
python_executable="$HOME/research/icl-dynamics/exp_001_hardware_check/venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
"$python_executable" -u close_at_checkpoint.py > logs/close_at_20m.log 2>&1
"$python_executable" -u audit_resume.py --checkpoint "$root/runs/main_resume_00002950016/checkpoints/00020000000.eqx" --output "$root/final_checkpoint_audit.json" > logs/final_checkpoint_audit.log 2>&1
"$python_executable" export_progress_curve.py --cutoff 20000000 > logs/final_curve_export.log 2>&1
"$python_executable" plot_results.py "$root/runs/main_resume_00002950016/log.h5" --prefix-log "$root/runs/main/log.h5" --max-sequences 20000000 --output "$root/final_00020000000" > logs/final_plot.log 2>&1
cp close_at_20m.json final_checkpoint_audit.json "$root/final_00020000000/"
cp runs/main/config.json "$root/final_00020000000/original_config.json"
cp runs/main_resume_00002950016/config.json "$root/final_00020000000/resume_config.json"
cp features/feature_manifest.json features/encoding_provenance.json "$root/final_00020000000/"
cp logs/train.log logs/train_resume_00002950016.log logs/final_checkpoint_audit.log "$root/final_00020000000/"
cp "$HOME/research/icl-dynamics/exp_001_hardware_check/source_manifest.json" "$root/final_00020000000/"
cp -r "$root/final_00020000000" /mnt/d/research/icl-dynamics/exp_002_main_reproduction/
date -u +%FT%TZ > finalization_20m.finished
