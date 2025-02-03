from segmenters.segmenter import Segmentor
from skimage.segmentation import slic
import torch

class Slic(Segmentor):
    def __init__(self) -> None:
        super().__init__()

    def get_segments(self, img_scikit, n_segments: int = 25):
        channel_axis = -1 if img_scikit.ndim == 3 else None
        segments_slic = slic(
            img_scikit,
            n_segments=n_segments,
            compactness=25.0,
            sigma=1.0,
            start_label=0,
            channel_axis=channel_axis,
        )
        return torch.from_numpy(segments_slic).unsqueeze(0)
