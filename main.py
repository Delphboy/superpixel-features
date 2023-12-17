import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet101_Weights, resnet101

from superpixel_features import (get_features_using_superpixels,
                                 get_superpixels, load_image)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = None

LOGGER = None

def get_logger(save_dir: str):
    global LOGGER
    if LOGGER is None:
        # set up logger
        logging.basicConfig(level=logging.INFO)
        # set logging file to log.txt
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"log-{save_dir.split('/')[-1]}.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        LOGGER = logger
    return LOGGER

logging.info("Device: {}".format(DEVICE))

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = resnet101(ResNet101_Weights.IMAGENET1K_V1)
        MODEL = nn.Sequential(*list(MODEL.children())[:-1])
        MODEL.eval()
        MODEL.to(DEVICE)
    return MODEL


def process(
    image_dir: str, output_dir: str, num_superpixels: int
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
        model = get_model()
        features = (
            get_features_using_superpixels(
                model=model, img=torch_image, super_pixel_masks=superpixels
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )

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

    args = args.parse_args()

    get_logger(args.save_dir)

    process(
        image_dir=args.image_dir,
        output_dir=args.save_dir,
        num_superpixels=args.num_superpixels,
    )
