import os

import numpy as np
import torch

feat_dim = 768


def get(image_locations, index):
    img_path = image_locations[index]
    image = np.load(img_path)["feat"]
    bbox = np.load(img_path)["bbox"]
    rag = np.load(img_path).get("rag", None)
    assert image.shape[0] == bbox.shape[0]
    image = torch.from_numpy(image)

    return image, bbox, rag


if __name__ == "__main__":
    # Get the files in test_output and store them in a list
    image_locations = os.listdir("test_out")
    image_locations = [os.path.join("test_out", i) for i in image_locations]

    total = 0.0
    for i in range(len(image_locations)):
        img, bbox, rag = get(image_locations, i)
        total += img.shape[0]
        print(
            f"{i+1}:\t",
            img.shape,
            bbox.shape,
            f"| {rag.shape}" if rag is not None else "",
        )
    print(f"Avg superpixels: {total/len(image_locations)}")
