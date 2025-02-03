import numpy as np
import torch
from skimage import graph


def create_rag_edges(image, superpixel_labels):
    image = np.transpose(image, (2, 0, 1))
    g = graph.rag_mean_color(image, superpixel_labels)
    edges = np.array(g.edges())  # [X, 2]

    edges = torch.tensor(edges).t().contiguous()
    edges = torch.cat([edges, torch.flip(edges, [0])], dim=1)
    edges = edges.cpu().numpy()
    return edges


def create_region_adjacency_graph(superpixels):
    # Initialize the edge list
    edge_list = []

    # Iterate over each pixel in the superpixel map
    for i in range(superpixels.shape[0]):
        for j in range(superpixels.shape[1]):
            # Get the current superpixel label
            current_label = superpixels[i, j]

            # Check the neighboring pixels
            for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # Calculate the neighboring pixel coordinates
                ni, nj = i + x, j + y

                # Check if the neighboring pixel is within the image boundaries
                if 0 <= ni < superpixels.shape[0] and 0 <= nj < superpixels.shape[1]:
                    # Get the neighboring superpixel label
                    neighbor_label = superpixels[ni, nj]

                    # If the neighboring superpixel is different, add an edge to the edge list
                    if neighbor_label != current_label:
                        edge_list.append([current_label, neighbor_label])

    # Remove duplicate edges
    edge_list = np.unique(edge_list, axis=0)

    # Convert the edge list to PyTorch tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index.cpu().numpy()

