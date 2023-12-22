#!/bin/bash

declare sets=("train" "val" "test")
declare sizes=(25 50 75 100)


for set in "${sets[@]}"
do
    for size in "${sizes[@]}"
    do
        echo "Processing $set with $size superpixels"
        sbatch --export=SET=$set,SIZE=$size submit.sh --job-name=$set-$size --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel-features/$set-$size.out submit.sh
    done
done

squeue -u $(whoami)