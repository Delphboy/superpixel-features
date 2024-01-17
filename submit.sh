#!/bin/bash

#SBATCH --chdir=/jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel-features
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=small

module load cuda/10.2
module load python/anaconda3

# source .venv/bin/activate
conda activate superpixels

SIZE=$(echo "$SIZE" | bc)

python3 main.py --image_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/coco/${SET}2014/ \
		--save_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/${SET}_m${SIZE} \
		--num_superpixels ${SIZE}

python3 merge_and_clean.py --input_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/${SET}_m${SIZE} \
							--output_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/m${SIZE}


# python3 main.py --image_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/coco/${SET}2014/ \
# 		--save_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/masked_${SET}_m${SIZE} \
# 		--num_superpixels ${SIZE} \
# 		--is_masked

# python3 merge_and_clean.py --input_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/masked_${SET}_m${SIZE} \
# 							--output_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/masked_m${SIZE}
