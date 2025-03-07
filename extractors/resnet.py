from extractors.extractor import Extractor

import torch
import torch.nn as nn
import torchvision.transforms as trans
from torchvision.models.resnet import ResNet101_Weights, resnet101

class ResNet(Extractor):
    def __init__(self) -> None:
        super().__init__()
        model = resnet101(ResNet101_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval()

        self.model.to(self._device)
        self.preprocess = ResNet101_Weights.IMAGENET1K_V1.transforms()

    def get_whole_img_features(self, img: torch.Tensor) -> torch.Tensor:
        img = self.preprocess(img).to(self._device)

        with torch.no_grad():
            features = self.model(img).squeeze(-1)
        return features.squeeze(-1).squeeze(0)

    def get_superpixel_features(self, superpixels: torch.Tensor, reshape: bool=True) -> torch.Tensor:
        """
        Given an image, create superpixel features using SLIC and ResNet101
        :param img: Image tensor of shape (b, c, h, w)
        :return: Tensor of superpixel features of shape (b, n_segments, 2048)
        """
        if reshape:
            superpixels = superpixels.reshape(-1, 3, 224, 224)
        superpixels = self.preprocess(superpixels)
        with torch.no_grad():
            features = self.model(superpixels).squeeze(-1)
        return features.squeeze(-1)
