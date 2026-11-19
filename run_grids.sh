#!/bin/bash
# Step 6 evaluation queue: attack grid and defense grid for every dataset and seed.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
log() { echo "[$(date +%H:%M)] $*" >> grids.log; }
run() { log "start $*"; $PY -u -m "$@" >> grids_jobs.log 2>&1 && log "done $*" || log "FAILED $*"; }

for ds in gtsrb cifar10; do
  for s in 0 1 2; do
    run securetinyml.eval.run_attacks $ds --seed $s --n 500
    run securetinyml.eval.run_defenses $ds --seed $s --n 500
  done
done
run securetinyml.eval.aggregate
run securetinyml.eval.charts
log "GRIDS COMPLETE"
