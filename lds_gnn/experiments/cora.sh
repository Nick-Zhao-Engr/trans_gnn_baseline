#!/bin/bash

runs=3
for i in $(seq 1 $runs);do
    python lds.py -m lds -e 0 -d cora -s -1;
done