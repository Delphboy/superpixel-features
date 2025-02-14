from extractors.extractor import Extractor
import torch
import torch.nn as nn
from lavis.models import load_model_and_preprocess

class Blip(Extractor):
    def __init__(self) -> None:
        super().__init__()

        self.model, vis_preprocess, _ = load_model_and_preprocess(
            name="blip_feature_extractor",
            model_type="base",
            is_eval=True,
            device=str(self._device),
        )
        self.preprocess = vis_preprocess["eval"]
        self.preprocess.transform.transforms.remove(self.preprocess.transform.transforms[1])

    def get_whole_img_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Given an image, create whole image features using BLIP
        :param img: Image tensor of shape (b, c, h, w)
        :return: Tensor of whole image features of shape (b, 768)
        """
        img = self.preprocess(img).to(self._device)
        sample = {"image": img}

        with torch.no_grad():
            features = self.model.extract_features(sample, mode="image")
        return features.image_embeds[0, 0, :]

    def get_superpixel_features(self, superpixels: torch.Tensor) -> torch.Tensor:
        """
        Given an image, create superpixel features using SLIC and ResNet101
        :param img: Image tensor of shape (b, c, h, w)
        :return: Tensor of superpixel features of shape (b, n_segments, 2048)
        """
        superpixels = superpixels.reshape(-1, 3, 224, 224)
        superpixels = self.preprocess(superpixels)
        sample = {"image": superpixels}
        with torch.no_grad():
            features = self.model.extract_features(sample, mode="image")

        return features.image_embeds[:, 0, :]

