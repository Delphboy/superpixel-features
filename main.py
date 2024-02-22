import argparse
import logging
import os

import numpy as np

from features import (
    get_blip_superpixel_features,
    get_blip_whole_img_features,
    get_clip_superpixel_features,
    get_clip_whole_img_features,
    get_resnet_patch_features,
    get_resnet_superpixel_features,
    get_resnet_whole_img_features,
)
from superpixels import get_patches, get_superpixels, load_image

LOGGER = None


def get_logger(save_dir: str):
    global LOGGER
    if LOGGER is None:
        # set up logger
        logging.basicConfig(level=logging.INFO)
        # set logging file to log.txt
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # handler = logging.FileHandler(f"log-{save_dir.split('/')[-1]}.txt")
        # handler.setLevel(logging.INFO)
        # formatter = logging.Formatter("%(message)s")
        # logger.addHandler(handler)
        LOGGER = logger
    return LOGGER


def process_superpixels(
    image_dir: str,
    output_dir: str,
    num_superpixels: int,
    is_masked: bool,
    model_id: str,
):
    # if output_dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the images in the directory
    images = os.listdir(image_dir)
    for i, image in enumerate(images):
        LOGGER.info(f"{i+1}/{len(images)} | Processing image: {image}")
        scikit_image, torch_image = load_image(os.path.join(image_dir, image))
        superpixels = get_superpixels(
            img_scikit=scikit_image, n_segments=num_superpixels
        )

        if model_id == "CLIP":
            features, bounding_boxes = get_clip_superpixel_features(
                img=torch_image,
                super_pixel_masks=superpixels,
                feat_resize_dim=512,
                is_masked=is_masked,
            )
        elif model_id == "BLIP":
            features, bounding_boxes = get_blip_superpixel_features(
                img=torch_image,
                super_pixel_masks=superpixels,
                feat_resize_dim=768,
                is_masked=is_masked,
            )
        else:
            features, bounding_boxes = get_resnet_superpixel_features(
                img=torch_image,
                super_pixel_masks=superpixels,
                feat_resize_dim=2048,
                is_masked=is_masked,
            )

        features = features.squeeze(0).cpu().numpy()
        bounding_boxes = bounding_boxes.squeeze(0).cpu().numpy()
        feats = {"feat": features, "bbox": bounding_boxes}
        np.savez_compressed(
            os.path.join(output_dir, image.split(".")[0] + ".npz"), **feats
        )


def process_patches(
    image_dir: str,
    output_dir: str,
    model_id: str,
):
    # if output_dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the images in the directory
    images = os.listdir(image_dir)
    for i, image in enumerate(images):
        LOGGER.info(f"{i+1}/{len(images)} | Processing image: {image}")
        scikit_image, torch_image = load_image(os.path.join(image_dir, image))
        patches = get_patches(img_torch=torch_image)

        if model_id == "CLIP":
            features, bounding_boxes = get_clip_superpixel_features(
                img=torch_image,
                super_pixel_masks=patches,
                feat_resize_dim=512,
                is_masked=False,
            )
        elif model_id == "BLIP":
            features, bounding_boxes = get_blip_superpixel_features(
                img=torch_image,
                super_pixel_masks=patches,
                feat_resize_dim=768,
                is_masked=False,
            )
        else:
            features = get_resnet_patch_features(
                patches=patches,
                feat_resize_dim=2048,
            )

        features = features.squeeze(0).cpu().numpy()
        feats = {"feat": features}
        np.savez_compressed(
            os.path.join(output_dir, image.split(".")[0] + ".npz"), **feats
        )


def process_whole_image(
    image_dir: str,
    output_dir: str,
    model_id: str,
):
    # if output_dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the images in the directory
    images = os.listdir(image_dir)
    for i, image in enumerate(images):
        LOGGER.info(f"{i+1}/{len(images)} | Processing image: {image}")
        _, torch_image = load_image(os.path.join(image_dir, image))

        if model_id == "CLIP":
            features = get_clip_whole_img_features(img=torch_image)
        elif model_id == "BLIP":
            features = get_blip_whole_img_features(img=torch_image)
        else:
            features = get_resnet_whole_img_features(img=torch_image)

        features = features.squeeze(0).cpu().detach().numpy()

        feats = {"feat": features}
        np.savez_compressed(
            os.path.join(output_dir, image.split(".")[0] + ".npz"), **feats
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--image_dir", type=str, required=True, help="Path to image dir")
    args.add_argument("--save_dir", type=str, required=True, help="Path to save dir")
    args.add_argument(
        "--num_superpixels",
        type=int,
        default=25,
        help="Number of superpixels to use",
    )
    args.add_argument(
        "--is_masked", action="store_true", help="Mask out non-superpixels?"
    )
    args.add_argument(
        "--model_id",
        type=str,
        default="resnet101",
        help="Which model to use? CLIP or resnet101",
    )
    args.add_argument(
        "--whole_img", action="store_true", help="Generate whole image features"
    )
    args.add_argument(
        "--patches",
        action="store_true",
        help="Generate patch features instead of superpixel features",
    )

    args = args.parse_args()

    get_logger(args.save_dir)

    if args.whole_img:
        process_whole_image(
            image_dir=args.image_dir,
            output_dir=args.save_dir,
            model_id=args.model_id,
        )
    elif args.patches:
        process_patches(
            image_dir=args.image_dir,
            output_dir=args.save_dir,
            model_id=args.model_id,
        )
    else:
        process_superpixels(
            image_dir=args.image_dir,
            output_dir=args.save_dir,
            num_superpixels=args.num_superpixels,
            is_masked=args.is_masked,
            model_id=args.model_id,
        )
