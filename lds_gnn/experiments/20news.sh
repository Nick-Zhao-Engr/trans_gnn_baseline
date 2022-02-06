
# runs=5
# for i in $(seq 1 $runs);do
#     python lds.py -m knnlds -e 0 -d 20news10 -s -1;
# done

# n_list=(1000 2000 3000 4000);
# for n in ${n_list[@]};do
#     python lds.py -m knnlds -e 0 -d 20news10 -node_remove $n
# done

python lds.py -m knnlds -e 0 -d 20news10 -node_remove 4000