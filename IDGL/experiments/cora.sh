#!/bin/bash

runs=5
# for i in $(seq 1 $runs);do
#     python main.py --config config/cora/idgl.yml
# done
python main.py --config config/cora/idgl.yml;
for i in $(seq 2 $runs);do
    python main.py --config config/cora/idgl${i}.yml
done 
