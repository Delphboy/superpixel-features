#!/bin/bash

declare sets=("train" "val" "test")
declare sizes=(15) #20 25 30 50 75 100)


for set in "${sets[@]}"
do
    # qsub -v SET=$set -N $set-whole submit_whole.qsub
    sbatch --export=SET=$set,SIZE=$size --job-name=$set-whole --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel-features/$set-whole.out submit_whole.sh

    for size in "${sizes[@]}"
    do
        echo "Processing $set with $size superpixels"
        sbatch --export=SET=$set,SIZE=$size --job-name=$set-$size --output=/jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel-features/$set-$size.out submit_super.sh
        # qsub -v SET=$set,SIZE=$size -N $set-$size submit_super.qsub
    done
done

# squeue -u $(whoami)