from typing import Optional

import clip
import torch
import torch.nn as nn
import torchvision.transforms as trans
from lavis.models import load_model_and_preprocess
from torchvision.models.resnet import ResNet101_Weights, resnet101

from superpixels import (
    _extract_masked_pixels_from_bounding_boxes,
    _extract_pixels_from_bounding_boxes,
    _get_bounding_boxes,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
preprocess = None


def _get_resnet():
    global model, preprocess
    if model is not None and preprocess is not None:
        return model, preprocess
    model = resnet101(ResNet101_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(DEVICE)
    preprocess = trans.Compose(
        [
            trans.Resize((224, 224)),
            trans.CenterCrop((224, 224)),
            trans.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return model, preprocess


def get_resnet_superpixel_features(
    img: torch.Tensor,
    super_pixel_masks: torch.Tensor,
    feat_resize_dim: int = 2048,
    is_masked: bool = False,
) -> torch.Tensor:
    """
    Given an image, create superpixel features using SLIC and ResNet101
    :param img: Image tensor of shape (b, c, h, w)
    :return: Tensor of superpixel features of shape (b, n_segments, 2048)
    """
    model, preprocess = _get_resnet()
    # Add batch dimension
    img = img.unsqueeze(0).to(DEVICE)
    super_pixel_masks = super_pixel_masks.unsqueeze(0).to(DEVICE)

    bounding_boxes = _get_bounding_boxes(img, super_pixel_masks)
    if is_masked:
        pixels = _extract_masked_pixels_from_bounding_boxes(
            img, bounding_boxes, super_pixel_masks
        ).to(DEVICE)
    else:
        pixels = _extract_pixels_from_bounding_boxes(img, bounding_boxes).to(DEVICE)

    pixels = pixels.reshape(-1, 3, 224, 224)
    pixels = preprocess(pixels)
    with torch.no_grad():
        features = model(pixels).squeeze(-1)
    features = features.reshape(-1, bounding_boxes.shape[1], feat_resize_dim)
    return features


def get_resnet_whole_img_features(
    img: torch.Tensor,
) -> torch.Tensor:
    """
    Given an image, create whole image features using ResNet101
    :param img: Image tensor of shape (b, c, h, w)
    :return: Tensor of whole image features of shape (b, 2048)
    """
    model, preprocess = _get_resnet()
    img = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model(img).squeeze(-1)
    return features


########################################################################################


def _get_clip():
    global model, preprocess
    if model is not None and preprocess is not None:
        return model, preprocess
    model_id = "ViT-B/32"
    model, _ = clip.load(model_id, device=DEVICE)
    preprocess = trans.Compose(
        [
            trans.Resize((224, 224)),
            trans.CenterCrop((224, 224)),
            trans.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    model.eval()
    model = model.encode_image
    return model, preprocess


def get_clip_superpixel_features(
    img: torch.Tensor,
    super_pixel_masks: torch.Tensor,
    feat_resize_dim: int = 2048,
    is_masked: bool = False,
) -> torch.Tensor:
    """
    Given an image, create superpixel features using SLIC and ResNet101
    :param img: Image tensor of shape (b, c, h, w)
    :return: Tensor of superpixel features of shape (b, n_segments, 2048)
    """
    model, preprocess = _get_clip()
    # Add batch dimension
    img = img.unsqueeze(0).to(DEVICE)
    super_pixel_masks = super_pixel_masks.unsqueeze(0).to(DEVICE)

    bounding_boxes = _get_bounding_boxes(img, super_pixel_masks)
    if is_masked:
        pixels = _extract_masked_pixels_from_bounding_boxes(
            img, bounding_boxes, super_pixel_masks
        ).to(DEVICE)
    else:
        pixels = _extract_pixels_from_bounding_boxes(img, bounding_boxes).to(DEVICE)

    pixels = pixels.reshape(-1, 3, 224, 224)
    pixels = preprocess(pixels)
    with torch.no_grad():
        features = model(pixels).squeeze(-1)
    features = features.reshape(-1, bounding_boxes.shape[1], feat_resize_dim)
    return features


def get_clip_whole_img_features(
    img: torch.Tensor,
) -> torch.Tensor:
    """
    Given an image, create whole image features using CLIP
    :param img: Image tensor of shape (b, c, h, w)
    :return: Tensor of whole image features of shape (b, 512)
    """
    model, preprocess = _get_clip()
    img = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model(img).squeeze(-1)
    return features


########################################################################################


def _get_blip():
    global model, preprocess
    if model is not None and preprocess is not None:
        return model, preprocess
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_feature_extractor",
        model_type="base",
        is_eval=True,
        device=DEVICE,
    )
    preprocess = vis_processors["eval"]
    preprocess.transform.transforms.remove(preprocess.transform.transforms[1])
    return model, preprocess


def get_blip_superpixel_features(
    img: torch.Tensor,
    super_pixel_masks: torch.Tensor,
    feat_resize_dim: int = 2048,
    is_masked: bool = False,
) -> torch.Tensor:
    """
    Given an image, create superpixel features using SLIC and ResNet101
    :param img: Image tensor of shape (b, c, h, w)
    :return: Tensor of superpixel features of shape (b, n_segments, 2048)
    """
    model, preprocess = _get_blip()
    # Add batch dimension
    img = img.unsqueeze(0).to(DEVICE)
    super_pixel_masks = super_pixel_masks.unsqueeze(0).to(DEVICE)

    bounding_boxes = _get_bounding_boxes(img, super_pixel_masks)
    if is_masked:
        pixels = _extract_masked_pixels_from_bounding_boxes(
            img, bounding_boxes, super_pixel_masks
        ).to(DEVICE)
    else:
        pixels = _extract_pixels_from_bounding_boxes(img, bounding_boxes).to(DEVICE)

    pixels = pixels.reshape(-1, 3, 224, 224)
    pixels = preprocess(pixels)
    sample = {"image": pixels}
    with torch.no_grad():
        features = model.extract_features(sample, mode="image")
    return features.image_embeds[:, 0, :].unsqueeze(0)


def get_blip_whole_img_features(
    img: torch.Tensor,
) -> torch.Tensor:
    """
    Given an image, create whole image features using BLIP
    :param img: Image tensor of shape (b, c, h, w)
    :return: Tensor of whole image features of shape (b, 768)
    """
    model, preprocess = _get_blip()
    img = preprocess(img).unsqueeze(0).to(DEVICE)
    sample = {"image": img}

    with torch.no_grad():
        features = model.extract_features(sample, mode="image")
    return features.image_embeds[0, 0, :].unsqueeze(0)
