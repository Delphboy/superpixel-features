from extractors.extractor import Extractor

import torch
import torch.nn as nn
import torchvision.transforms as trans
from torchvision.models.resnet import ResNet101_Weights, resnet101
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

class Vit(Extractor):
    def __init__(self) -> None:
        super().__init__()
        self.model = vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(self._device)

        self.preprocess = ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    def get_whole_img_features(self, img: torch.Tensor) -> torch.Tensor:
        img = self.preprocess(img).to(self._device)

        with torch.no_grad():
            features = self.model(img)
        return features.squeeze(0)

    def get_superpixel_features(self, superpixels: torch.Tensor, reshape:bool=True) -> torch.Tensor:
        """
        Given an image, create superpixel features using superpixels and ViT B 16
        :param img: Image tensor of shape (b, c, h, w)
        :return: Tensor of superpixel features of shape (b, n_segments, 2048)
        """
        if reshape:
            superpixels = superpixels.reshape(-1, 3, 224, 224)
        superpixels = self.preprocess(superpixels)
        with torch.no_grad():
            features = self.model(superpixels)
        return features
