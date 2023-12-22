from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as trans
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float

transforms = trans.Compose([trans.ToTensor()])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path: str):
    """
    Load an image from a path and return it as a scikit image and a torch image
    :param image_path: Path to image
    :return: scikit image and torch image
    """
    img = Image.open(image_path)
    img_scikit = img_as_float(img)

    img_torch = transforms(img)
    return img_scikit, img_torch


def _run_slic(
    img,
    n_segments: Optional[int] = 25,
    compactness: Optional[float] = 10.0,
    sigma: Optional[float] = 1.0,
    start_label: Optional[int] = 0,
):
    segments_slic = slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=start_label,
    )
    return segments_slic


def get_superpixels(img_scikit, n_segments: Optional[int] = 25):
    """
    Get the superpixels of an image using SLIC
    :param img_scikit: scikit image
    :param n_segments: Number of superpixels
    :return: Superpixel segmentation
    """
    segments_slic = _run_slic(img_scikit, n_segments=n_segments)
    return torch.from_numpy(segments_slic)


def _get_bounding_boxes(img: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """
    Given an image and its superpixel segmentation, create bounding boxes
    for each superpixel
    :param img: Image tensor of shape (b, c, h, w)
    :param seg: Superpixel segmentation tensor of shape (b, h, w)
    :return: Tensor of bounding boxes of shape (b, max_seg, 4)
    """
    B, H, W = seg.shape
    bounding_boxes = torch.zeros((B, seg.max() + 1, 4)).to(img.device)

    for b in range(B):
        for s in range(seg.max() + 1):
            # Get the indices of the superpixel
            indices = torch.where(seg[b] == s)

            if len(indices[0]) == 0:
                continue

            # Get the bounding box of the superpixel
            x_min, y_min = torch.min(indices[1]), torch.min(indices[0])
            x_max, y_max = torch.max(indices[1]), torch.max(indices[0])

            # Get the pixels of the superpixel
            if x_max - x_min == 0 or y_max - y_min == 0:
                continue
            bounding_boxes[b, s] = torch.tensor([x_min, y_min, x_max, y_max])

    return bounding_boxes


def _extract_pixels_from_bounding_boxes(
    img: torch.Tensor, bounding_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Extract the pixels from the image that are in the bounding boxes
    :param img: Image tensor of shape (b, c, h, w)
    :param bounding_boxes: Tensor of bounding boxes of shape (b, max_seg, 4) where the bounding boxes are in the format (x_min, y_min, x_max, y_max)
    :return: Tensor of pixels of shape (b, max_seg, 3, 224, 224)
    """
    B, C, H, W = img.shape
    max_seg = bounding_boxes.shape[1]
    pixels = torch.zeros((B, max_seg, 3, 224, 224)).to(img.device)

    for b in range(B):
        for s in range(max_seg):
            x_min, y_min, x_max, y_max = bounding_boxes[b, s]
            if x_max - x_min == 0 or y_max - y_min == 0:
                continue

            pixels[b, s] = F.interpolate(
                img[
                    b, :, y_min.int() : y_max.int(), x_min.int() : x_max.int()
                ].unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

    return pixels


def _extract_masked_pixels_from_bounding_boxes(
    img: torch.Tensor, bounding_boxes: torch.Tensor, seg: torch.Tensor
) -> torch.Tensor:
    """
    Extract the pixels from the image that are in the bounding boxes, masking out the pixels that are not in the superpixel
    :param img: Image tensor of shape (b, c, h, w)
    :param bounding_boxes: Tensor of bounding boxes of shape (b, max_seg, 4) where the bounding boxes are in the format (x_min, y_min, x_max, y_max)
    :param seg: Superpixel segmentation tensor of shape (b, h, w)
    :return: Tensor of pixels of shape (b, max_seg, 3, 224, 224)
    """
    max_seg = bounding_boxes.shape[1]
    new_img = torch.zeros_like(img.unsqueeze(1).repeat(1, max_seg, 1, 1, 1))
    B, max_seg, C, H, W = new_img.shape
    pixels = torch.zeros((B, max_seg, 3, 224, 224)).to(img.device)

    for b in range(B):
        for s in range(max_seg):
            indices = torch.where(seg[b] == s)
            if len(indices[0]) == 0:
                continue
            # mask the image so that only the pixels in the superpixel are visible
            new_img[b, s, :, indices[0], indices[1]] = img[b, :, indices[0], indices[1]]

            x_min, y_min, x_max, y_max = bounding_boxes[b, s]
            if x_max - x_min == 0 or y_max - y_min == 0:
                continue

            pixels[b, s] = F.interpolate(
                new_img[
                    b, s, :, y_min.int() : y_max.int(), x_min.int() : x_max.int()
                ].unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

    return pixels


def get_features_using_superpixels(
    model: torch.nn.Module,
    img: torch.Tensor,
    super_pixel_masks: torch.Tensor,
    is_masked: bool = False,
) -> torch.Tensor:
    """
    Given an image, create superpixel features using SLIC and ResNet101
    :param img: Image tensor of shape (b, c, h, w)
    :return: Tensor of superpixel features of shape (b, n_segments, 2048)
    """
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
    with torch.no_grad():
        features = model(pixels).squeeze(-1)
    features = features.reshape(-1, bounding_boxes.shape[1], 2048)
    return features
