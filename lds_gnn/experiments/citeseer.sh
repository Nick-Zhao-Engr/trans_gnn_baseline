#!/bin/bash

runs=5
for i in $(seq 1 $runs);do
    python lds.py -m lds -e 0 -d citeseer -s -1;
done