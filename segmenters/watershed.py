from enum import unique
import torch
from segmenters.segmenter import Segmentor
from skimage.segmentation import watershed


class Watershed(Segmentor):
    def __init__(self) -> None:
        super().__init__()

    def get_segments(self, img_scikit, n_segments: int = 25):
        segments = watershed(img_scikit, markers=n_segments, compactness=1)
        # segments = segments - 1
        if len(segments.shape) == 3:
            segments = segments[:, :, 0]

        return torch.from_numpy(segments).unsqueeze(0)
