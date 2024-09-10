import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, GCNConv
from torch_geometric.utils import to_networkx


def pd(msg: str):
    print(f"DEBUG | Visualise | {msg}")


def _run_conv(graph: Data):
    network = GCNConv(2048, 512).cpu()
    x = network(graph.x.cpu(), graph.edge_index.cpu())


def _build_nx_graph(image, superpixels, features, edges=None):
    if edges is None:
        edges = knn_graph(features, k=3)

    pyg_graph = Data(x=features.contiguous(), edge_index=edges)
    pd(f"Superpixels {np.unique(superpixels.reshape(-1))}")
    pd(f"Graph {pyg_graph}")
    pd(f"Nodes {pyg_graph.x.shape}")
    pd(f"Edges {pyg_graph.edge_index.shape}")
    _run_conv(pyg_graph)
    # pd(f"Edges:{edges}")

    G = to_networkx(pyg_graph, to_undirected=False)

    return G


def visualise_graph(save_loc, image, superpixels, features, edges=None):
    G = _build_nx_graph(image, superpixels, features, edges)
    fig, ax = plt.subplots(figsize=(50, 50))

    ax.imshow(image, alpha=0.9)
    # ax.contour(superpixels, linewidths=1, colors="yellow")

    # Calculate node positions for the figure
    positions = {}
    for i in np.unique(superpixels.reshape(-1)):
        mask = superpixels == i
        center_x = np.median(np.where(mask)[1])
        center_y = np.median(np.where(mask)[0])

        positions[i] = (center_x, center_y)

    # Create a color map for the superpixels
    cmap = plt.get_cmap("jet")
    num_superpixels = len(np.unique(superpixels))
    colors = [cmap(i / num_superpixels) for i in range(num_superpixels)]

    # Create a single overlay for all superpixels
    overlay = np.zeros((*superpixels.shape, 3))
    for i, color in enumerate(colors):
        mask = superpixels == i
        overlay[mask] = color[:3]  # Use only the RGB values

    ax.imshow(overlay, alpha=0.4)

    # Debug circles
    # for k, pos in positions.items():
    #     pd(pos)
    #     c = plt.Circle(pos, 20, color="black")
    #     ax.text(
    #         pos[0], pos[1], f"{k}", ha="center", va="center", color="white", fontsize=36
    #     )
    #     ax.add_patch(c)

    pd(f"The are {len(positions)} positions and {len(G.nodes)} nodes")

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

    fig.suptitle("DEBUGGING", fontsize=75)
    # fig.suptitle("10 SLIC superpixels - BLIP RAG", fontsize=75)
    plt.tight_layout()
    plt.savefig(save_loc + "-visualisation.png")
