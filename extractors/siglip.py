from extractors.extractor import Extractor
import clip
import torch
import torchvision.transforms as trans
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

class Siglip(Extractor):
    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(self._device)
        self.preprocess = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    def get_whole_img_features(self, img: torch.Tensor) -> torch.Tensor:
        input = self.preprocess(images=img.to(self._device), 
                                text=["placeholder text"], 
                                padding="max_length", 
                                return_tensors="pt")

        for k, v in input.items():
            input[k] = v.to(self._device)

        with torch.no_grad():
            features = self.model(**input)

        map = features['vision_model_output'].pooler_output
        x = features['vision_model_output'].last_hidden_state
        return map.squeeze(0)


    def get_superpixel_features(self, superpixels: torch.Tensor, reshape:bool=True) -> torch.Tensor:
        if reshape:
            superpixels = superpixels.reshape(-1, 3, 224, 224)
        preprocessed_items = self.preprocess(images=superpixels.to(self._device),
                                      text=['placehold test'],
                                      padding="max_length",
                                      return_tensors="pt")

        for k, v in preprocessed_items.items():
            preprocessed_items[k] = v.to(self._device)

        with torch.no_grad():
            features = self.model(**preprocessed_items)

        map = features['vision_model_output'].pooler_output
        return map

