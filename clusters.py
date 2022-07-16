from matplotlib.pyplot import grid
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.feature_extraction.image import grid_to_graph
import torch

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# We fit data, we fit labeled data, and we label clusters
def FeatureAgglomerationClustering( vit_output: torch.Tensor, n_clusters: int):
    # Creating clustering tree
    bs, z, x, y = vit_output.shape
    vit_output = vit_output.view( -1, vit_output.shape[1]).detach().cpu()

    connectivity = grid_to_graph(x, y ,z )
    cluster1 = FeatureAgglomeration(n_clusters=100, compute_distances=True)
    cluster1 = cluster1.fit(vit_output)
    plot_dendrogram(cluster1, truncate_mode="level", p=4)
    plt.savefig("dum2.png")


if __name__ == '__main__':
    x = torch.randn(size = (32, 21, 12, 12)).cuda()
    FeatureAgglomerationClustering(x, 21)