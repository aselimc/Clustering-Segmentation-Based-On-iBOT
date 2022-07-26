import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchpq.clustering import KMeans
from tqdm import tqdm

from utils import extract_feature, mIoU


class KMeansSegmentator(nn.Module):

    def __init__(self, backbone, logger,
                 k=20,
                 num_classes=21,
                 feature='intermediate',
                 patch_labeling='coarse',
                 n_blocks=1,
                 use_cuda=True,
                 distance='euclidean',
                 **kwargs):
        super(KMeansSegmentator, self).__init__()

        self.use_cuda = use_cuda
        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.backbone = backbone.to(device=self.device)
        self.feature = feature
        self.n_blocks = n_blocks
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        self.num_classes = num_classes
        self.patch_labeling = patch_labeling

        self.kmeans = KMeans(n_clusters=k, distance=distance, **kwargs)

        self.logger = logger

    @torch.no_grad()
    def forward(self, image):
        bs = image.size(0)
        feat = extract_feature(self.backbone, image, feature=self.feature, n_blocks=self.n_blocks)
        feat = feat.flatten(start_dim=0, end_dim=1).permute(1, 0).contiguous()
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

        return pred

    @torch.no_grad()
    def fit(self, loader):
        train_features = []
        train_labels = []
        self.cluster_labels = []

        print("Extracting ViT features...")
        progress_bar = tqdm(total=len(loader))
        for image, target in loader:
            image = image.to(device=self.device)
            feat = extract_feature(self.backbone, image, feature=self.feature, n_blocks=self.n_blocks)
            feat = feat.flatten(start_dim=0, end_dim=1).cpu()
            train_features.append(feat)

            # divide ground truth mask into patches
            target = target.to(device=self.device)
            target = self.unfold(target.unsqueeze(1).float())
            target = target.permute(0, 2, 1)
            if self.patch_labeling == 'coarse':
                target = target.long()
                target = F.one_hot(target, self.num_classes)
                target = torch.argmax(target.sum(dim=2), dim=2)
                target = target.unsqueeze(2).expand(-1, self.num_patches, self.patch_size**2)
            target = target.flatten(start_dim=0, end_dim=1)
            target = target.byte().cpu()
            train_labels.append(target)
            progress_bar.update()
        
        train_features = torch.cat(train_features, dim=0).permute(1, 0).contiguous()
        train_labels = torch.cat(train_labels, dim=0).long()
        train_labels = F.one_hot(train_labels, self.num_classes)

        # fit clusters, i.e. get centroids (embed_dim, k)
        print("\nFitting clusters...")
        self.kmeans.fit(train_features)

        # label clusters
        print("Assigning cluster labels...")
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

    @torch.no_grad()
    def score(self, loader):
        top1 = []
        
        print("Compute score...")
        progress_bar = tqdm(total=len(loader))
        for idx, (image, target) in enumerate(loader):
            image = image.to(device=self.device)
            target = target.to(device=self.device)

            pred = self.forward(image)
            top1.append(mIoU(pred, target))

            if idx % self.logger.config['eval_freq'] == 0 or idx == len(loader):
                self.logger.log_segmentation(image[0], pred[0], target[0], step=idx, logit=False)
            progress_bar.update()

        top1 = torch.stack(top1)
        miou = torch.mean(top1).item()
        iou_std = torch.std(top1).item()

        self.logger.log_scalar_summary({
                "mIoU": miou,
                "IoU std": iou_std,
            })

        return miou, iou_std

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

    @property
    def embed_dim(self):
        return self.backbone.embed_dim

    @property
    def patch_size(self):
        return self.backbone.patch_embed.patch_size

    @property
    def img_size(self):
        return self.backbone.patch_embed.img_size

    @property
    def num_patches(self):
        return int((self.img_size / self.patch_size)**2)
