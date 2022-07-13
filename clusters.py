from matplotlib.pyplot import grid
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.feature_extraction.image import grid_to_graph
import torch

def FeatureAgglomerationClustering( classifier_output: torch.Tensor, n_classes: int):
    # Creating clustering tree
    bs, z, x, y = classifier_output.shape
    classifier_output = classifier_output.view(-1, classifier_output.shape[1]).detach().cpu()

    connectivity = grid_to_graph(x, y ,z )
    cluster1 = FeatureAgglomeration(n_clusters=2, connectivity=connectivity)
    cluster1.fit(classifier_output)
    x = 5


if __name__ == '__main__':
    x = torch.randn(size = (32, 21, 6, 6)).cuda()
    FeatureAgglomerationClustering(x, 21)