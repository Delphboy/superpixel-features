import math

import torch
import torch.nn.functional as F
from segmenters.segmenter import Segmentor

import torchvision.transforms as trans

class Patcher(Segmentor):
    def __init__(self) -> None:
        super().__init__()
        self._HEIGHT = 224
        self._WIDTH = 224

    def _calculate_patch_size(self, n_segments: int) -> int:
        """
        Given an image of dimension 224x224, patching into 16x16 patches will give
        (224 * 224) / (16 * 16) = 196 patches
        Therefore: n_segments = (H * W) / (k^2) 
        With the assumption that patches will be square
        To find k, rearrange:
        k = sqrt((H * W) / n_segments)
        """
        kernel_size = math.sqrt((self._HEIGHT * self._WIDTH) / n_segments)
        assert kernel_size % 1 == 0, f"BAD SEGMENT COUNT: num_segments={n_segments} will yield a patching kernel size of {kernel_size}. Kernel size must be an integer"
        return int(kernel_size)


    def get_segments(self, img_scikit, n_segments: int = 25):
        img_torch = torch.from_numpy(img_scikit).to(self._device)
        if len(img_torch.shape) != 3:
            img_torch = img_torch.unsqueeze(2).repeat(1, 1, 3)
        img_torch = img_torch.unsqueeze(0).permute(0, 3, 1, 2)

        preprocess = trans.Compose(
            [
                trans.Resize((self._HEIGHT, self._WIDTH)),
                trans.CenterCrop((self._HEIGHT, self._WIDTH)),
            ]
        )
        img_torch = preprocess(img_torch)

        patch_size = self._calculate_patch_size(n_segments)

        patches = F.unfold(img_torch, kernel_size=(patch_size, patch_size), stride=patch_size)
        patches = patches.permute(0, 2, 1)
        patches = patches.reshape(-1, 3, patch_size, patch_size).to(torch.float32)
        return patches
