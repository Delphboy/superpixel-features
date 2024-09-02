import torch
import torch.nn.functional as F
import torchvision.transforms as trans
from PIL import Image
from skimage.segmentation import slic, watershed
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
    # handle edge case where torch_img only has 1 channel
    if img_torch.shape[0] == 1:
        img_torch = img_torch.repeat(3, 1, 1)
    return img_scikit, img_torch


def _get_bounding_boxes(img: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """
    Given an image and its superpixel segmentation, create bounding boxes
    for each superpixel
    :param img: Image tensor of shape (b, c, h, w)
    :param seg: Superpixel segmentation tensor of shape (b, h, w)
    :return: Tensor of bounding boxes of shape (b, max_seg, 4)
    """
    B, H, W = seg.shape
    bounding_boxes = torch.zeros((B, seg.max() - 1 + 1, 4)).to(img.device)

    for b in range(B):
        for s in range(seg.max()):
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

    assert (
        max(seg.reshape(-1).unique()) == bounding_boxes.shape[1]
    ), "The number of superpixels and bounding boxes does not match"
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


def _run_slic(
    img,
    n_segments: int = 25,
    compactness: float = 25.0,
    sigma: float = 1.0,
    start_label: int = 0,
):
    channel_axis = -1 if img.ndim == 3 else None
    segments_slic = slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=start_label,
        channel_axis=channel_axis,
    )
    return segments_slic


def _run_watershed(img, n_segments: int = 25):
    segments = watershed(img, markers=n_segments, compactness=0.001)
    # Convert to 0-based indexing
    segments = segments - 1
    return segments


def get_superpixels(img_scikit, n_segments: int = 25, algo: str = "SLIC"):
    """
    Get the superpixels of an image using SLIC
    :param img_scikit: scikit image
    :param n_segments: Number of superpixels
    :return: Superpixel segmentation
    """
    if algo == "SLIC":
        segments = _run_slic(img_scikit, n_segments=n_segments)  # [X,Y]
    elif algo == "watershed":
        segments = _run_watershed(img_scikit, n_segments=n_segments)  # [X,Y,C]
        if len(segments.shape) == 3:
            segments = segments[:, :, 0]
    else:
        raise ValueError(f"Algorithm {algo} not supported.")

    return torch.from_numpy(segments)


def get_patches(img_torch):
    """
    Get 16x16 patches of an image
    :param img_torch: torch image
    :return: Patches
    """
    img_torch = img_torch.to(DEVICE)
    img_torch = img_torch.unsqueeze(0)

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
    img_torch = preprocess(img_torch)

    patches = F.unfold(img_torch, kernel_size=16, stride=16)
    patches = patches.permute(0, 2, 1)
    patches = patches.reshape(-1, 3, 16, 16)
    return patches
