import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm

from . import _BaseSegmentator
from utils import extract_feature, mIoU


class KNNSegmentator(_BaseSegmentator):

    def __init__(self, backbone,
                 logger,
                 k=20,
                 weighted_majority_vote=False,
                 temperature=1.0,
                 **kwargs):
        """
        Args:
            feature: 'intermediate', 'query', 'key', 'value'
            patch_label: 'coarse', 'fine'
        """
        super(KNNSegmentator, self).__init__(backbone, logger, **kwargs)

        self.k = k
        self.weighted_majority_vote = weighted_majority_vote
        self.temperature = temperature

    @torch.no_grad()
    def forward(self, image):
        bs = image.size(0)
        test_feature = self._extract_feature(image, flatten=False)

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
        pred = self.smooth(pred)

        return pred

    @torch.no_grad()
    def fit(self, loader):
        train_features, train_labels = self._transform_data(loader)
        train_features, train_labels = self._balance_class(train_features, train_labels)

        self.train_features = train_features.permute(1, 0)
        self.train_labels = train_labels.permute(1, 0)

    @torch.no_grad()
    def score(self, loader):
        self.train_features = self.train_features.to(device=self.device)
        self.train_labels = self.train_labels.to(device=self.device)
        top1 = []

        progress_bar = tqdm(total=len(loader))
        for idx, (image, target) in enumerate(loader):
            image = image.to(device=self.device)
            target = target.to(device=self.device)

            pred = self.forward(image)
            bs = pred.size(0)
            for i in range(bs):
                top1.append(mIoU(pred[i], target[i]))

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
