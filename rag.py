import torch
import numpy as np
from skimage import graph

def create_rag_edges(image, superpixel_labels):
    g = graph.rag_mean_color(image, superpixel_labels)
    edges = np.array(g.edges()) # [X, 2]

    # import pytorch geometric and modify the edges so that they are "undirected"
    edges = torch.tensor(edges).t().contiguous()
    edges = torch.cat([edges, torch.flip(edges, [0])], dim=1)
    edges = edges.cpu().numpy()
    return edges
