import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchpq.clustering import KMeans
from tqdm import tqdm

from . import _BaseSegmentator
from utils import mIoU
from utils.transforms import PatchwiseSmoothMask


class KMeansSegmentator(_BaseSegmentator):

    def __init__(self, backbone, logger,
                 k=20,
                 distance='euclidean',
                 init_mode='kmeans++',
                 n_redo=10,
                 max_iter=300,
                 tol=1e-4,
                 **kwargs):
        super(KMeansSegmentator, self).__init__(backbone, logger, **kwargs)

        self.kmeans = KMeans(n_clusters=k,
                             distance=distance,
                             init_mode=init_mode,
                             n_redo=n_redo,
                             max_iter=max_iter,
                             tol=tol)

    @torch.no_grad()
    def forward(self, image):
        bs = image.size(0)
        feat = self._extract_feature(image)
        feat = feat.permute(1, 0).contiguous()
        feat = feat.cpu()

        cluster_assignment = self.kmeans.predict(feat)
        cluster_assignment = cluster_assignment.to(device=self.device)
        cluster_assignment = cluster_assignment.view(bs, self.num_patches).unsqueeze(2)
        cluster_assignment = cluster_assignment.expand(bs, self.num_patches, self.patch_size**2).unsqueeze(3)
        cluster_labels = self.cluster_labels.expand(bs, self.num_patches, self.patch_size**2, self.k)
        patch_preds = torch.gather(cluster_labels, dim=3, index=cluster_assignment)
        patch_preds = patch_preds.view(bs, self.num_patches, 1, self.patch_size, self.patch_size)

        # tile label patches together -> (bs, k, 224, 224)
        nrows = self.img_size // self.patch_size
        pred = [make_grid(patch_pred, nrows, padding=0)[0] for patch_pred in patch_preds]
        pred = torch.stack(pred)
        pred = self.smooth(pred)

        return pred

    @torch.no_grad()
    def fit(self, loader):
        train_features, train_labels = self._transform_data(loader)
        train_features = train_features.permute(1, 0).contiguous()
        train_labels = train_labels.long()
        train_labels = F.one_hot(train_labels, self.num_classes)

        # fit clusters, i.e. get centroids (embed_dim, k)
        print("\nFitting clusters...")
        self.kmeans.fit(train_features)

        # label clusters
        print("Assigning cluster labels...")
        self.cluster_labels = []
        cluster_assignment = self.kmeans.predict(train_features)
        train_features = train_features.to(device=self.device)
        train_labels = train_labels.to(device=self.device)

        similarities = self._similarity(train_features, self.centroids.to(device=self.device))
        for idx in range(self.k):
            # weighted majority vote accross patches, higher similarity -> higher weight
            assigned_similarities = similarities[cluster_assignment == idx, idx]
            weights = torch.softmax(assigned_similarities, dim=0)
            weights = weights * 0.0 + 1.0

            assigned_train_labels = train_labels[cluster_assignment == idx]
            vote = torch.sum(weights[:, None, None] * assigned_train_labels, dim=0)
            label = torch.argmax(vote, dim=1)
            self.cluster_labels.append(label)

        self.cluster_labels = torch.stack(self.cluster_labels, dim=1).unsqueeze(0).unsqueeze(0)

    def _similarity(self, x, y, inplace=False, normalize=True):
        return self.kmeans.sim(x, y, inplace, normalize)

    @property
    def centroids(self):
        return self.kmeans.centroids
    
    @property
    def k(self):
        return self.kmeans.n_clusters

    @property
    def distance(self):
        return self.kmeans.distance
