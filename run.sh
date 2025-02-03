#!/bin/bash

conda activate sp

img_root="./test_inp/"
out_root="./test_out/"
model_id="RESNET"
algo='SLIC'

rm ${out_root}/${model_id}/${algo}/*

python3 new_main.py --image_dir ${img_root} \
        --save_dir ${out_root}/${model_id}/${algo} \
        --feature_extractor ${model_id} \
        --segmenter ${algo} \
        --num_segments 14
echo ""
ls ${out_root}/${model_id}/${algo} \

echo "\n\n"
python3 merge_and_clean.py --save_dir ${out_root}/${model_id}/${algo} \
        --output_dir ${out_root}/${model_id}/${algo} 
echo ""
ls ${out_root}/${model_id}/${algo} \
