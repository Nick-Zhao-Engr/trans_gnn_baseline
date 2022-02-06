#!/bin/bash

runs=3
for i in $(seq 1 $runs);do
    python lds.py -m knnlds -e 0 -d mini;
done