#!/bin/bash

conda init
conda activate super

declare sets=("train" "val" "test")
declare sizes=(15 50)
declare algos=("SLIC")

img_root="/home/hsenior/coco/img"
out_root="/home/hsenior/coco/superpixel_features"
model_id="BLIP"

for set in "${sets[@]}"
do
    # Generate whole image features
    python3 main.py --image_dir ${img_root}/${set}2014/ \
            --save_dir ${out_root}/${model_id}/whole_img \
            --feature_extractor ${model_id} \
            --whole_img

    python3 merge_and_clean.py --save_dir ${out_root}/${model_id}/whole_img \
                                --output_dir ${out_root}/${model_id}/whole_img \


    for algo in "${algos[@]}"
    do
        for size in "${sizes[@]}"
        do
            echo "Processing $set with $size superpixels using $algo"
            # Generate superpixel features
            python3 main.py --image_dir ${img_root}/${set}2014/ \
                    --save_dir ${out_root}/${model_id}/${algo}/${set}_m${size} \
                    --num_superpixels ${size} \
                    --algorithm ${algo} \
                    --feature_extractor ${model_id} \
                    --rag

            python3 merge_and_clean.py --save_dir ${out_root}/${model_id}/${algo}/${set}_m${size} \
                                        --output_dir ${out_root}/${model_id}/${algo}/m${size} \
                                        --delete
        done
    done
done
