import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm

from utils import extract_feature, mIoU


class KNNSegmentator(nn.Module):

    def __init__(self, backbone,
                 logger,
                 k=20,
                 num_classes=21,
                 feature='intermediate',
                 patch_labeling='coarse',
                 background_label_percentage=0.2,
                 weighted_majority_vote=False,
                 n_blocks=1,
                 temperature=1.0,
                 use_cuda=True):
        """
        Args:
            feature: 'intermediate', 'query', 'key', 'value'
            patch_label: 'coarse', 'fine'
        """
        super(KNNSegmentator, self).__init__()

        self.backbone = backbone
        self.n_blocks = n_blocks
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        self.logger = logger

        self.k = k
        self.num_classes = num_classes
        self.feature = feature
        self.patch_labeling = patch_labeling
        self.background_label_percentage = background_label_percentage
        self.weighted_majority_vote = weighted_majority_vote
        self.temperature = temperature

        self.use_cuda = use_cuda
        if use_cuda:
            self.backbone.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    @torch.no_grad()
    def forward(self, image):
        bs = image.size(0)
        test_feature = extract_feature(self.backbone, image, feature=self.feature, n_blocks=self.n_blocks)

        # patchwise cosine similarity between test feature & all train features
        # (bs x num_patches x embed_dim) * (embed_dim x (num_patches * num_train))
        similarity = torch.matmul(test_feature, self.train_features)
        similarity, indices = similarity.topk(self.k, largest=True, sorted=True)
        indices = indices.unsqueeze(2).expand(bs, self.num_patches, self.patch_size**2, self.k)
        train_labels = self.train_labels.unsqueeze(0).unsqueeze(0)
        train_labels = train_labels.expand(bs, self.num_patches, self.patch_size**2, -1)
        retrieved_neighbors = torch.gather(train_labels, dim=3, index=indices)

        # tile label patches together -> (bs, k, 224, 224)
        nrows = ncols = self.img_size // self.patch_size
        retrieved_neighbors = retrieved_neighbors.permute(0, 1, 3, 2)
        retrieved_neighbors = retrieved_neighbors.view(bs, self.num_patches, self.k, self.patch_size, self.patch_size)
        retrieved_neighbors = [make_grid(nns, nrows, padding=0) for nns in retrieved_neighbors]
        retrieved_neighbors = torch.stack(retrieved_neighbors)

        if self.weighted_majority_vote:
            # more similarity = higher voting weight
            similarity = similarity.permute(0, 2, 1).view(bs, self.k, nrows, ncols)
            similarity = F.interpolate(similarity,
                                    size=[self.img_size, self.img_size],
                                    mode='nearest',
                                    recompute_scale_factor=False)
            similarity = similarity.permute(0, 2, 3, 1).unsqueeze(-1) / self.temperature
            similarity = torch.softmax(similarity, dim=3)
        else:
            similarity = 1.0

        # voting
        retrieved_neighbors = retrieved_neighbors.permute(0, 2, 3, 1).long()
        retrieved_neighbors = F.one_hot(retrieved_neighbors, self.num_classes)
        vote = (similarity * retrieved_neighbors).sum(dim=3)
        pred = torch.argmax(vote, dim=-1)

        return pred

    @torch.no_grad()
    def fit(self, loader):
        train_features, train_labels = [], []

        progress_bar = tqdm(total=len(loader))
        for image, target in loader:
            image = image.to(device=self.device)
            feat = extract_feature(self.backbone, image, feature=self.feature, n_blocks=self.n_blocks)
            feat = feat.flatten(start_dim=0, end_dim=1)
            feat = feat.cpu()
            train_features.append(feat)

            # divide ground truth mask into patches
            target = self.unfold(target.unsqueeze(1).float())
            if self.patch_labeling == 'coarse':
                target = target.permute(0, 2, 1).long()
                target = F.one_hot(target, self.num_classes)
                target = torch.argmax(target.sum(dim=2), dim=2)
                target = target.unsqueeze(2).expand(-1, self.num_patches, self.patch_size**2)
            else:
                target = target.permute(0, 2, 1)
            target = target.flatten(start_dim=0, end_dim=1)
            target = target.byte().cpu()
            train_labels.append(target)

            progress_bar.update()

        self.train_features = torch.cat(train_features, dim=0)
        self.train_labels = torch.cat(train_labels, dim=0)

        # class balancing
        is_background = (self.train_labels == 0).all(dim=1)
        idx_background = torch.nonzero(is_background, as_tuple=False).squeeze(1).numpy()
        idx_foreground = torch.nonzero(~is_background, as_tuple=False).squeeze(1).numpy()
        size_undersample = int(len(idx_background) * self.background_label_percentage)
        idx_undersample = np.random.choice(size_undersample, size=size_undersample, replace=False)
        idx_train = np.concatenate([idx_foreground, idx_undersample], axis=0)

        self.train_features = self.train_features[idx_train].permute(1, 0)
        self.train_labels = self.train_labels[idx_train].permute(1, 0)

    @torch.no_grad()
    def score(self, loader):
        self.train_features = self.train_features.to(device=self.device)
        self.train_labels = self.train_labels.to(device=self.device)
        top1 = []

        progress_bar = tqdm(total=len(loader))
        for idx, (image, target) in enumerate(loader):
            image = image.to(device=self.device)
            target = target.to(device=self.device)

            preds = self.forward(image)
            for pred in preds:
                top1.append(mIoU(pred, target))

            if idx % self.logger.config['eval_freq'] == 0 or idx == len(loader):
                self.logger.log_segmentation(image[0], preds[0], target[0], step=idx, logit=False)
            progress_bar.update()

        top1 = torch.stack(top1)
        miou = torch.mean(top1).item()
        iou_std = torch.std(top1).item()

        self.logger.log_scalar_summary({
                "mIoU": miou,
                "IoU std": iou_std,
            })

        return miou, iou_std

    @property
    def patch_size(self):
        return self.backbone.patch_embed.patch_size

    @property
    def img_size(self):
        return self.backbone.patch_embed.img_size

    @property
    def num_patches(self):
        return int((self.img_size / self.patch_size)**2)
