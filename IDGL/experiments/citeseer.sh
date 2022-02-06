#!/bin/bash

runs=4
for i in $(seq 1 $runs);do
    python main.py --config config/citeseer/idgl.yml
done