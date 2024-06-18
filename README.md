# Superpixel Features

Generate features for superpixels and patches using pretrained models. Features are saves as `.npz` files.

## Dependencies

- PyTorch
- PyTorch Geometric
- Scikit Learn Image
- OpenAI CLIP
- Salesforce LAVIS

## Parameters

| Name | Description |
|--|--|
| `--image_dir` | The directory containnig image inputs |
| `--save_dir` | The directory to save the `npz` files to |
| `--num_superpixels` | The number of superpixels to generate per image |
| `--model_id` | Which model to use? [BLIP / CLIP / ResNet] |
| `--whole_img` | (Flag) Generate a single feature for the whole image |
| `--is_masked` | (Flag) Black out pixels in the superpixel bounding box that aren't in the original superpixel |
| `--patches` | (Flag) Generate patch features instead of superpixel features |

> [!WARNING]
> The `--patches` flag will generate $16 \times 16$ patches for an image that is resized to $224 \times 224$, yielding $14 \times 14 = 196$ patches

## Example

```bash
python3 main.py --image_dir "/homes/hps01/superpixel-features/test_images" \
    --save_dir "/homes/hps01/superpixel-features/test_output/" \
    --is_masked \
    --model_id "BLIP" \
    --num_superpixels 25 \
```
