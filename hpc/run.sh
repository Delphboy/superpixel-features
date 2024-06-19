#!/bin/bash

declare sets=("train" "val" "test")
declare sizes=(15 25 30 50 75)
declare algos=("SLIC" "watershed")

for set in "${sets[@]}"
do
    # qsub -v SET=$set -N $set-whole submit_whole.qsub

    for algo in "${algos[@]}"
    do
        for size in "${sizes[@]}"
        do
            echo "Processing $set with $size superpixels using $algo"
            qsub -v SET=$set,SIZE=$size,ALGO=$algo -N $set-$size-$algo submit_super.qsub
        done
    done
done

sleep 5

qstat
