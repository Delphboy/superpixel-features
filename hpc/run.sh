#!/bin/bash

declare sets=("train" "val" "test")
declare sizes=(15) #20 25 30 50 75 100)
declare algos=("slic" "watershed")

for algo in "${algos[@]}"
do
    for set in "${sets[@]}"
    do
        qsub -v SET=$set,ALGO=$algo -N $set-whole submit_whole.qsub

        for size in "${sizes[@]}"
        do
            echo "Processing $set with $size superpixels using $algo"
            qsub -v SET=$set,SIZE=$size,ALGO=$algo -N $set-$size submit_super.qsub
        done
    done
done

# squeue -u $(whoami)