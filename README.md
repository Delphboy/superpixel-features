# Superpixel Features

Generate features for superpixels and patches using pretrained models. Features are saved as `.npz` files with keys for the features (`'feats'`), superpixel bounding box (`'bbox'`), and region adjacency edges (`'rag'`).

In terms of feature space, the code supports ResNet, CLIP, and BLIPv2 features. Currently, only SLIC and Watershed superpixel segmentation algorithms are implemented (via scikit-image) however, patching is also supported (but not with the `--rag` flag).

To generate a collection of superpixels for the COCO dataset, see `runner.sh` for an example of how this can be achieved.

For compatiblity with the Karpathy Split of the COCO dataset, `merge_and_clean.py` is provided. This script will move and rename the superpixel feature files such that they can be used in place of the BUTD files in the original Karpathy Split JSON files. 

## Dependencies

```bash
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install scikit-image
python3 -m pip install salesforce-lavis
python3 -m pip install clip-openai
```

## Parameters

| Name | Description |
|--|--|
| `--image_dir` | The directory containnig image inputs |
| `--save_dir` | The directory to save the `npz` files to |
| `--feature_extractor` | Which model to use? [BLIP / CLIP / ResNet] |
| `--num_superpixels` | The number of superpixels to generate per image (Not compatible with `--whole_img`) |
| `--algorithm` | Which superpixel algorithm to use? [SLIC / watershed] |
| `--whole_img` | (Flag) Generate a single feature for the whole image (Not compatible with `--rag`) |
| `--patches` | (Flag) Generate patch features instead of superpixel features (Not compatible with `--rag`) |
| `--rag` | (Flag) Generate the Region Adjacency Graph edges between superpixels |

> [!WARNING]
> The `--patches` flag will generate $16 \times 16$ patches for an image that is resized to $224 \times 224$, yielding $14 \times 14 = 196$ patches

## Examples

Generate Watershed superpixel CLIP features for the Karpathy Test Set

```bash
python3 main.py --image_dir "/home/hsenior/coco/img/test2014/" \
    --save_dir "/home/hsenior/coco/superpixel_features/" \
    --model_id "CLIP" \
    --num_superpixels 25 \
    --algorithm "watershed" \
    --rag
```

Generate whole image ResNet features for the Karpathy Validation set

```bash
python3 main.py --image_dir "/home/hsenior/coco/img/val2014/" \
    --save_dir "/home/hsenior/coco/superpixel_features/" \
    --model_id "CLIP" \
    --whole_img
```

Generate SLIC superpixel features for the Karpathy Train Set (without the RAG edges)

```bash
python3 main.py --image_dir "/home/hsenior/coco/img/train2014/" \
    --save_dir "/home/hsenior/coco/superpixel_features/" \
    --model_id "BLIP" \
    --num_superpixels 75 \
    --algorithm "SLIC" \
```
