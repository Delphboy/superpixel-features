import os

import numpy as np
import torch

feat_dim = 768


def get(image_locations, index):
    img_path = image_locations[index]
    image = np.load(img_path)["feat"]
    image = torch.from_numpy(image)

    # if image.shape[0] < 50 then we need to pad it with zeros
    if image.shape[0] < 50:
        pad = torch.zeros((50 - image.shape[0], feat_dim))
        image = torch.cat((image, pad), 0)

    return image[:50]


if __name__ == "__main__":
    # Get the files in test_output and store them in a list
    image_locations = os.listdir("test_output")
    image_locations = [os.path.join("test_output", i) for i in image_locations]

    for i in range(len(image_locations)):
        assert get(image_locations, i).shape == (50, feat_dim)
