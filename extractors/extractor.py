import torch
from torch._prims_common import DeviceLikeType

class Extractor:
    def __init__(self) -> None:
        self._device: DeviceLikeType = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_whole_img_features(self, img:torch.Tensor) -> torch.Tensor:
        pass

    def get_superpixel_features(self, superpixels: torch.Tensor, reshape:bool=True) -> torch.Tensor:
        """
        Given the pixels from the bounding boxes surrounding the superpixel masks, generate a feature vector
        :param superpixels (torch.Tensor): A [B, n_segments, 3, 224, 224] tensor
        :return: torch.Tensor [B, n_segments, D]
        """
        pass

