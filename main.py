from extractors.clip import Clip
from extractors.blip import Blip
from extractors.extractor import Extractor
from extractors.resnet import ResNet
from extractors.siglip import Siglip
from extractors.vit import Vit
from segmenters.patcher import Patcher
from segmenters.segmenter import Segmentor
from segmenters.slic import Slic
from segmenters.watershed import Watershed
from rag import create_rag_edges
from visualise import visualise_graph

import os
import logging
import argparse

import numpy as np
import torch
import torchvision.transforms as trans
from PIL import Image
from skimage.util import img_as_float

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

SEG_SLIC = "SLIC"
SEG_WATERSHED = "WATERSHED"
SEG_PATCHER = "PATCHER"

SUPPORTED_SEGS = [SEG_SLIC, SEG_WATERSHED, SEG_PATCHER]

MODEL_RESNET = "RESNET"
MODEL_CLIP = "CLIP"
MODEL_BLIP = "BLIP"
MODEL_SIGLIP = "SIGLIP"
MODEL_VIT = "VIT"

SUPPORTED_MODELS = [MODEL_RESNET, MODEL_CLIP, MODEL_BLIP, MODEL_SIGLIP, MODEL_VIT]

def build_segmenter(args):
    segmenter_type = str(args.segmenter)
    if segmenter_type == SEG_SLIC:
        return Slic()
    elif segmenter_type == SEG_WATERSHED:
        return Watershed()
    elif segmenter_type == SEG_PATCHER:
        return Patcher()
    else:
        return Segmentor()


def build_extractor(args):
    model_name = str(args.feature_extractor)
    if model_name == MODEL_RESNET:
        return ResNet()
    elif model_name == MODEL_CLIP:
        return Clip()
    elif model_name == MODEL_BLIP:
        return Blip()
    elif model_name == MODEL_SIGLIP:
        return Siglip()
    elif model_name == MODEL_VIT:
        return Vit()
    else:
        return Extractor()


def sanity_check_args(args):
    model_name = str(args.feature_extractor)#.upper()
    seg_name = str(args.segmenter)
    is_patcher = seg_name == SEG_PATCHER
    is_visualised = args.visualise
    is_rag = args.rag
    is_whole_img = args.whole_img
    is_partial_segmenter = bool(args.segmenter) ^ bool(args.num_segments)
    is_legal_segmenter = bool((bool(args.segmenter) and args.num_segments))

    assert model_name in SUPPORTED_MODELS, f"UNKNOWN EXTRACTOR: The model selected ({model_name}) is not supported. Please use one of {SUPPORTED_MODELS}"
    assert is_whole_img or seg_name in SUPPORTED_SEGS, f"UNKNOWN SEGMENTER: The segmenter selected ({seg_name}) is not supported. Please use one of {SUPPORTED_SEGS}"

    # Option checks
    assert not(is_whole_img and is_partial_segmenter), "OPTIONS MISMATCH: Must generate whole image features or have a legal segmentation (segmenter and num_segments) configuration. ILLEGAL SEGMENTER DETECTED"
    assert is_whole_img ^ is_legal_segmenter, "OPTIONS MISMATCH: Must generate whole image features or segmentation features"

    assert not((is_rag and is_patcher) or (args.rag and args.whole_img)), "OPTIONS MISMATCH: Cannot generate RAG with patches or whole image features."
    assert not((not is_rag) and is_visualised), "OPTIONS MISMATCH: Cannot visualise unless building a RAG"
    assert is_whole_img ^ (is_legal_segmenter and args.num_segments > 0), f"BAD SEGMENT COUNT: num_segments must be > 0. Received {args.num_segments}"

    # Directory checks
    assert os.path.exists(args.image_dir), f"MISSING INPUT DIRECTORY: Image directory {args.image_dir} not found"
    assert len(os.listdir(args.image_dir)), f"EMPTY INPUT DIRECTORY: Image directory {args.image_dir} is empty"
    if not os.path.exists(args.save_dir):
        logging.warning(f"Creating save directory: {str(args.save_dir)}")
        os.makedirs(str(args.save_dir))


def load_image(image_path: str):
    """
    Load an image from a path and return it as a scikit image and a torch image
    :param image_path: Path to image
    :return: scikit image and torch image
    """
    img = Image.open(image_path)
    img_scikit = img_as_float(img)

    transforms = trans.Compose([trans.ToTensor()])
    img_torch: torch.Tensor = transforms(img)
    if img_torch.shape[0] == 1:
        img_torch = img_torch.repeat(3, 1, 1)
    return img_scikit, img_torch.unsqueeze(0)


def save_features(location, name, feat_dict):
    np.savez_compressed(os.path.join(location, name) + ".npz", **feat_dict)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--image_dir", type=str, required=True, help="Path to image dir")
    args.add_argument("--save_dir", type=str, required=True, help="Path to save dir")

    args.add_argument("--feature_extractor", type=str, default="CLIP", help="Which model to use for feature extraction?")

    args.add_argument("--whole_img", action="store_true", help="Generate whole image features")
    args.add_argument("--segmenter",type=str, help="Superpixel algorithm to use")
    args.add_argument("--num_segments", type=int, help="Number of superpixels to use")

    args.add_argument("--rag", action="store_true", help="Add RAG edge features to the superpixel features")
    args.add_argument("--visualise", action="store_true", help="Produce a visualisation of the superpixels")

    args = args.parse_args()
    sanity_check_args(args)

    segmenter = build_segmenter(args)
    extractor = build_extractor(args)

    for i, image_file in enumerate(os.listdir(args.image_dir)):
        LOGGER.info(f"{i+1} / {len(os.listdir(args.image_dir))}\t{image_file}")

        # Load Images
        image_path = os.path.join(args.image_dir, image_file)
        img_scikit, img_torch = load_image(image_path)
        feat_dict = {}

        if not args.whole_img and type(segmenter) is not Patcher:
            # Superpixel features
            masks = segmenter.get_segments(img_scikit, args.num_segments)
            bboxes = segmenter._get_bounding_boxes(img_torch, masks)
            feat_dict['bbox'] = bboxes.squeeze(0).cpu().numpy()
            pixels = segmenter._extract_pixels_from_bounding_boxes(img_torch, bboxes) # [B, n_segments, 3, 224, 224]
            feats = extractor.get_superpixel_features(pixels)

            if args.rag:
                feat_dict['rag'] = create_rag_edges(img_scikit, masks.cpu().numpy())

            if args.visualise:
                visualise_graph(
                    os.path.join(args.save_dir, image_file.split(".")[0]),
                    img_scikit,
                    masks,
                    feats,
                    torch.tensor(feat_dict['rag']),
                )

        elif type(segmenter) is Patcher:
            # Patch features
            patches = segmenter.get_segments(img_scikit, args.num_segments)
            feats = extractor.get_superpixel_features(patches, reshape=False)
        else:
            # Whole image feature
            feats = extractor.get_whole_img_features(img_torch)

        feats = feats.cpu().numpy()
        feat_dict['feat'] = feats

        save_features(args.save_dir, image_file.split('.')[0], feat_dict)


