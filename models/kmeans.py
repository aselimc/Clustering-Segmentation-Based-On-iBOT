import numpy as np
from sklearn import cluster
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm

from utils import IoU


def l2_distance(input, centroids, dim=1):
    return torch.pow(centroids - input, 2).sum(dim=dim)


class KMeans(nn.Module):

    def __init__(self,
                 k=20,
                 embed_dim=1024,
                 max_iter=300,
                 tol=1e-4,
                 n_init=10,
                 centroid_init='random',
                 use_cuda=True):
        super(KMeans, self).__init__()

        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init

        self.use_cuda = use_cuda
        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.centroid_init = centroid_init
        self.embed_dim = embed_dim
        self.centroids = torch.zeros(embed_dim, k).to(self.device)

    @torch.no_grad()
    def predict(self, test_features):
        test_features = test_features.to(device=self.centroids.device)
        num_samples = test_features.size(0)

        centroids = self.centroids.unsqueeze(0).expand(num_samples, self.embed_dim, -1)
        l2_dists = l2_distance(test_features.unsqueeze(2), centroids)
        assignment = torch.argmin(l2_dists, dim=1)

        return assignment

    @torch.no_grad()
    def fit(self, train_features):
        train_features = train_features.to(device=self.device)
        
        if self.centroid_init == 'kmeans++':
            self._kmeans_plusplus(train_features)
        else:
            num_samples = train_features.size(0)
            centroid_idx = np.random.choice(num_samples, size=self.k, replace=False)
            self.centroids = train_features[centroid_idx].transpose(0, 1)

        print("\nFit clusters...")
        progress_bar = tqdm(total=self.max_iter)
        for it in range(self.max_iter):
            new_centroids = self._step(train_features)

            centroid_similarity = (self.centroids * new_centroids).sum(dim=0)
            centroid_change = torch.abs(centroid_similarity).sum()
            if centroid_change < self.tol:
                progress_bar.update(self.max_iter - it)
                break

            self.centroids = new_centroids

            progress_bar.update()

    def _step(self, train_features):
        new_centroids = torch.zeros_like(self.centroids).to(device=self.device)
        num_samples = train_features.size(0)
        centroids = self.centroids.unsqueeze(0).expand(num_samples, self.embed_dim, self.k)

        # (bs x num_patches x embed_dim) * (embed_dim x k)
        l2_dists = l2_distance(train_features.unsqueeze(2), centroids)
        assignment = torch.argmin(l2_dists, dim=1)

        for idx in range(self.k):
            new_centroids[:, idx] += train_features[assignment == idx].mean(dim=0)

        return new_centroids

    def _kmeans_plusplus(self, train_features):
        train_features = train_features
        num_samples = train_features.size(0)

        first_centroid_idx = np.random.choice(num_samples)
        img_idx = first_centroid_idx // self.num_patches
        patch_idx = first_centroid_idx % self.num_patches
        first_centroid = train_features[img_idx, patch_idx]
        self.centroids[:, 0] = first_centroid

        for i in range(1, self.k):
            sampled_centroids = self.centroids[:, :i]
            sampled_centroids = sampled_centroids.unsqueeze(0).expand(num_samples, self.embed_dim, -1)
            l2_dists = l2_distance(train_features.unsqueeze(2), sampled_centroids)
            l2_dists, _ = torch.max(l2_dists, dim=1)

            # random sampling of 2 + log(k) many centroid candidates
            # lower similarity -> higher probability  
            rand_vals = torch.rand(size=(2+int(np.log(self.k)),), device=l2_dists.device)
            rand_vals *= l2_dists.sum()
            cdf = l2_dists.cumsum(dim=0)
            candidate_idx = torch.searchsorted(cdf, rand_vals)

            candidate_centroids = train_features[candidate_idx]
            candidate_l2_dists = l2_dists[candidate_idx]
            idx = torch.argmax(candidate_l2_dists, dim=0)
            centroid = candidate_centroids[idx]
            self.centroids[:, i] = centroid


class KMeansSegmentator(KMeans):

    def __init__(self, backbone, logger,
                 num_classes=21,
                 feature='intermediate',
                 patch_labeling='coarse',
                 n_blocks=1,
                 **kwargs):
        super(KMeansSegmentator, self).__init__(embed_dim=backbone.embed_dim, **kwargs)

        self.backbone = backbone.to(device=self.device)
        self.feature = feature
        self.n_blocks = n_blocks
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        self.num_classes = num_classes
        self.patch_labeling = patch_labeling

        self.logger = logger

    def forward(self, image):
        bs = image.size(0)
        feat = self._extract_feature(image).unsqueeze(3)

        centroids = self.centroids.unsqueeze(0).unsqueeze(0)
        centroids = centroids.expand(bs, self.num_patches, self.embed_dim, self.k)

        l2_dists = l2_distance(feat, centroids, dim=2)
        assignment = torch.argmax(l2_dists, dim=2)
        assignment = assignment.unsqueeze(2).expand(bs, self.num_patches, self.patch_size**2).unsqueeze(3)
        cluster_labels = self.cluster_labels.expand(bs, self.num_patches, self.patch_size**2, self.k)
        patch_preds = torch.gather(cluster_labels, dim=3, index=assignment)
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
            feat = self._extract_feature(image)
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
        
        train_features = torch.cat(train_features, dim=0)
        train_labels = torch.cat(train_labels, dim=0).long()
        train_labels = F.one_hot(train_labels, self.num_classes)

        # fit clusters, i.e. get centroids (embed_dim, k)
        super(KMeansSegmentator, self).fit(train_features)

        # label clusters
        # weighted majority vote accross patches, higher distance -> lower weight
        cluster_assignment = super(KMeansSegmentator, self).predict(train_features)
        train_features = train_features.to(device=self.device)
        train_labels = train_labels.to(device=self.device)
        for idx in range(self.k):
            num_assignments = (cluster_assignment == idx).sum()
            centroid = self.centroids[:, idx]
            centroid = centroid.unsqueeze(0).expand(num_assignments, self.embed_dim)
            assigned_train_features = train_features[cluster_assignment == idx]
            assigned_train_labels = train_labels[cluster_assignment == idx]
            
            l2_dists = l2_distance(assigned_train_features, centroid)
            weights = torch.softmax(l2_dists, dim=0)
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
            top1.append(IoU(pred, target))

            progress_bar.update()

            if idx % self.logger.config['eval_freq'] == 0 or idx == len(loader):
                self.logger.log_segmentation(image[0], pred[0], target[0], step=idx, logit=False)
            progress_bar.update()

        top1 = torch.cat(top1, dim=0)
        miou = torch.mean(top1).item()
        iou_std = torch.std(top1).item()

        self.logger.log_scalar_summary({
                "mIoU": miou,
                "IoU std": iou_std,
            })

        return miou, iou_std

    def _extract_feature(self, images):
        if self.feature == 'intermediate':
            intermediate_output = self.backbone.get_intermediate_layers(images, self.n_blocks)
        else:
            intermediate_output = self.backbone.get_qkv(images, self.n_blocks, out=self.feature)
        feat = torch.stack(intermediate_output, dim=2)
        feat = torch.mean(feat, dim=2)
        feat = feat[:, 1:]

        return feat

    @property
    def patch_size(self):
        return self.backbone.patch_embed.patch_size

    @property
    def img_size(self):
        return self.backbone.patch_embed.img_size

    @property
    def num_patches(self):
        return int((self.img_size / self.patch_size)**2)
