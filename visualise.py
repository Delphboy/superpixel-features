import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_networkx


def pd(msg: str):
    print(f"DEBUG | Visualise | {msg}")


def _build_nx_graph(image, superpixels, features, edges=None):
    if edges is None:
        edges = knn_graph(features, k=15)

    pyg_graph = Data(x=features.contiguous(), edge_index=edges)

    G = to_networkx(pyg_graph, to_undirected=False)

    return G


def visualise_graph(save_loc, image, superpixels, features, edges=None):
    G = _build_nx_graph(image, superpixels, features, edges)
    fig, ax = plt.subplots(figsize=(50, 50))

    ax.imshow(image, alpha=0.8)
    ax.contour(superpixels, linewidths=5, colors="yellow")

    positions = {}
    for i in np.unique(superpixels.reshape(-1)):
        mask = superpixels == i
        center_x = np.mean(np.where(mask)[1])
        center_y = np.mean(np.where(mask)[0])

        positions[i] = (center_x, center_y)

    nx.draw_networkx(
        G,
        ax=ax,
        pos=positions,
        node_size=7000,
        font_size=32,
        width=5,
        arrowsize=50,
        arrowstyle="->",
        # connectionstyle="arc3,rad=0.1",
    )

    fig.suptitle("10 SLIC superpixels - BLIP RAG", fontsize=75)
    plt.tight_layout()
    plt.savefig(save_loc + "-visualisation.png")
