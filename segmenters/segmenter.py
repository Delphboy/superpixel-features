import torch
import torch.nn.functional as F
import torchvision.transforms as trans
from PIL import Image
from skimage.util import img_as_float

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Segmentor:
    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = trans.Compose([trans.ToTensor()])

    def load_image(self, image_path: str):
        """
        Load an image from a path and return it as a scikit image and a torch image
        :param image_path: Path to image
        :return: scikit image and torch image
        """
        img = Image.open(image_path)
        img_scikit = img_as_float(img)

        img_torch: torch.Tensor = self.transforms(img)
        if img_torch.shape[0] == 1:
            img_torch = img_torch.repeat(3, 1, 1)
        return img_scikit, img_torch.unsqueeze(0)

    def _get_bounding_boxes(self, img: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """
        Given an image and its superpixel segmentation, create bounding boxes
        for each superpixel
        :param img: Image tensor of shape (b, c, h, w)
        :param seg: Superpixel segmentation tensor of shape (b, h, w)
        :return: Tensor of bounding boxes of shape (b, max_seg, 4)
        """
        B, H, W = seg.shape
        seg_count = len(seg.unique())
        bounding_boxes = torch.zeros((B, seg_count, 4)).to(img.device)

        for b in range(B):
            for s in range(seg_count):
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
            len(seg.reshape(-1).unique()) == bounding_boxes.shape[1]
        ), f"SUPERPIXEL BBOX SHAPE MISMATCH: The number of superpixels ({len(seg.reshape(-1).unique())}) and bounding boxes ({bounding_boxes.shape[1]}) does not match"
        return bounding_boxes

    def _extract_pixels_from_bounding_boxes(self, img: torch.Tensor, bounding_boxes: torch.Tensor) -> torch.Tensor:
        """
        Extract the pixels from the image that are in the bounding boxes
        :param img: Image tensor of shape (b, c, h, w)
        :param bounding_boxes: Tensor of bounding boxes of shape (b, max_seg, 4) where the bounding boxes are in the format (x_min, y_min, x_max, y_max)
        :return: Tensor of pixels of shape (b, max_seg, 3, 224, 224)
        """
        B = img.shape[0]
        max_seg = bounding_boxes.shape[1]
        pixels = torch.zeros((B, max_seg, 3, 224, 224)).to(img.device)

        for b in range(B):
            for s in range(max_seg):
                x_min, y_min, x_max, y_max = bounding_boxes[b, s]
                if x_max - x_min == 0 or y_max - y_min == 0:
                    continue

                pixels[b, s] = F.interpolate(
                    img[b, :, y_min.int() : y_max.int(), x_min.int() : x_max.int()].unsqueeze(0),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )

        return pixels.to(self._device)


    def get_segments(self, img_scikit, n_segments: int = 25):
        raise NotImplementedError("get_superpixels() has not been implemented")

