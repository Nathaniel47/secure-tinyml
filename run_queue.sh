#!/bin/bash
# Step 6 training queue (laptop CPU, one job at a time). Each job resumes from its checkpoint.
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe
log() { echo "[$(date +%H:%M)] $*" >> queue.log; }
run() { log "start $*"; $PY -u -m "$@" >> queue_jobs.log 2>&1 && log "done $*" || log "FAILED $*"; }

for s in 1 2; do run securetinyml.models.train cifar10 --epochs 10 --seed $s; done
for s in 1 2; do run securetinyml.defenses.adv_train cifar10 --epochs 6 --eps 4 --seed $s; done
for s in 0 1 2; do run securetinyml.models.train gtsrb --epochs 6 --seed $s; done
for s in 0 1 2; do run securetinyml.defenses.adv_train gtsrb --epochs 4 --eps 4 --seed $s; done
for r in 0.01 0.02; do run securetinyml.models.train cifar10 --epochs 8 --seed 0 --poison-rate $r; done
run securetinyml.models.train gtsrb --epochs 6 --seed 0 --poison-rate 0.05
log "QUEUE COMPLETE"
