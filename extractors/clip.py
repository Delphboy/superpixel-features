from extractors.extractor import Extractor
import clip
import torch
import torchvision.transforms as trans

class Clip(Extractor):
    def __init__(self) -> None:
        super().__init__()

        model_id = "ViT-B/32"
        model, _ = clip.load(model_id, device=str(self._device))
        model.eval()

        self.model = model.encode_image
        self.preprocess = trans.Compose(
            [
                trans.Resize((224, 224)),
                trans.CenterCrop((224, 224)),
                trans.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def get_whole_img_features(self, img: torch.Tensor) -> torch.Tensor:
        img = self.preprocess(img).to(self._device)
        with torch.no_grad():
            features = self.model(img).squeeze(-1).squeeze(0)
        return features

    def get_superpixel_features(self, superpixels: torch.Tensor) -> torch.Tensor:
        superpixels = superpixels.reshape(-1, 3, 224, 224)
        superpixels = self.preprocess(superpixels)
        with torch.no_grad():
            features = self.model(superpixels).squeeze(-1)

        return features

    def get_patch_features(self, patches) -> torch.Tensor:
        patches = self.preprocess(patches).to(self._device)
        with torch.no_grad():
            features = self.model(patches).squeeze(-1)
        return features

