import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def _build_nx_graph(features, edges):
    pyg_graph = Data(x=features.contiguous(), edge_index=edges)
    G = to_networkx(pyg_graph, to_undirected=False)

    return G


def visualise_graph(save_loc, image, superpixels, features, edges=None):
    G = _build_nx_graph(features, edges)
    fig, ax = plt.subplots(figsize=(50, 50))

    ax.imshow(image, alpha=0.9)

    # Calculate node positions for the figure
    positions = {}
    for i in np.unique(superpixels.reshape(-1)):
        mask = superpixels == i
        # print(np.where(mask))
        centre_x = np.median(np.where(mask)[2])
        centre_y = np.median(np.where(mask)[1])

        positions[i] = (centre_x, centre_y)

    # Create a color map for the superpixels
    cmap = plt.get_cmap("jet")
    num_superpixels = len(np.unique(superpixels))
    colors = [cmap(i / num_superpixels) for i in range(num_superpixels)]

    # Create a single overlay for all superpixels
    overlay = np.zeros((*superpixels.shape, 3))
    for i, color in enumerate(colors):
        mask = superpixels == i
        overlay[mask] = color[:3]  # Use only the RGB values

    ax.imshow(overlay.squeeze(0), alpha=0.4)

    nx.draw_networkx(
        G,
        ax=ax,
        pos=positions,
        node_size=10000,
        font_size=32,
        width=5,
        arrowsize=50,
        arrowstyle="->",
        alpha=0.7,
        # connectionstyle="arc3,rad=0.1",
    )

    fig.suptitle("Region Adjacency Graph", fontsize=75)
    plt.tight_layout()
    plt.savefig(save_loc + "-visualisation.png")
