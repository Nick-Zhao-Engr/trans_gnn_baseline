#!/usr/bin/env bash

# Runs an example of VGCN on CiteSeer in the no-graph case
DATASET='citeseer' # ['citeseer', 'cora']
MC_SAMLES_TEST=3 # We used 16 in the paper
PRIOR_TYPE='smoothing'  # No-graph case: uses KNNG to build prior (ignoring the original)
RESULTS_DIR='./results/citeseer' # Results for every epoch will be saved here
EPOCH=2000

PYTHONPATH=. python experiments/run_vgcn.py --prior-type=$PRIOR_TYPE --posterior-type=free --num-epochs=$EPOCH --initial-learning-rate=0.001 --mc-samples-train=3 --mc-samples-test=3 --dropout-rate=0.5 --layer-type=dense  --log-every-n-iter=50 --zero-smoothing-factor=1e-5 --relaxed --fixed-split --dataset-name=$DATASET  --l2-reg-scale-gcn=5e-4  --one-smoothing-factor=0.9  --temperature-prior=0.1  --temperature-posterior=0.1  --beta=1  --seed=1 --seed-val=3 --seed-np=1 --results-dir=$RESULTS_DIR VGCN

