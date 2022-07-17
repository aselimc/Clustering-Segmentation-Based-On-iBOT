import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm

from utils import IoU


class KNNSegmentator(nn.Module):

    def __init__(self, backbone,
                 logger,
                 k=20,
                 num_classes=21,
                 feature='intermediate',
                 n_blocks=1,
                 use_cuda=True):
        """
        Args:
            feature: 'intermediate', 'query', 'key', 'value'
        """
        super(KNNSegmentator, self).__init__()

        self.backbone = backbone
        self.n_blocks = n_blocks
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        self.logger = logger

        self.k = k
        self.num_classes = num_classes
        self.feature = feature

        self.use_cuda = use_cuda
        if use_cuda:
            self.backbone.cuda()

    def forward(self, image):
        bs = image.size(0)
        test_feature = self.extract_feature(image).unsqueeze(3)

        # patchwise cosine similarity between test feature & all train features
        # (bs x num_patches x embed_dim x 1) * (num_patches x embed_dim x num_train)
        similarity = (test_feature * self.train_features).sum(dim=2)
        similarity, indices = similarity.topk(self.k, largest=True, sorted=True)
        indices = indices.unsqueeze(2).expand(bs, self.num_patches, self.patch_size**2, self.k)
        train_labels = self.train_labels.unsqueeze(0).expand(bs, self.num_patches, self.patch_size**2, -1)
        retrieved_neighbors = torch.gather(train_labels, dim=3, index=indices)

        # tile label patches together -> (bs, k, 224, 224)
        nrows = ncols = self.img_size // self.patch_size
        retrieved_neighbors = retrieved_neighbors.permute(0, 1, 3, 2)
        retrieved_neighbors = retrieved_neighbors.view(bs, self.num_patches, self.k, self.patch_size, self.patch_size)
        retrieved_neighbors = [make_grid(nns, nrows, padding=0) for nns in retrieved_neighbors]
        retrieved_neighbors = torch.stack(retrieved_neighbors)

        # weights for majority vote (more similarity = higher voting weight)
        similarity = similarity.permute(0, 2, 1).view(bs, self.k, nrows, ncols)
        similarity = F.interpolate(similarity,
                                   size=[self.img_size, self.img_size],
                                   mode='nearest',
                                   recompute_scale_factor=False)
        similarity = similarity.permute(0, 2, 3, 1).unsqueeze(-1)
        similarity = torch.softmax(similarity, dim=3)

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
            if self.use_cuda:
                image = image.cuda()
            feat = self.extract_feature(image)
            feat = feat.cpu()
            train_features.append(feat)

            # divide ground truth mask into patches
            target = self.unfold(target.unsqueeze(1).float())
            target = target.permute(2, 1, 0)
            target = target.byte().cpu()
            train_labels.append(target)

            progress_bar.update()

        self.train_features = torch.cat(train_features, dim=0).permute(1, 2, 0)
        self.train_labels = torch.cat(train_labels, dim=2)

    @torch.no_grad()
    def score(self, loader):
        if self.use_cuda:
            self.train_features = self.train_features.cuda()
            self.train_labels = self.train_labels.cuda()

        top1 = []

        progress_bar = tqdm(total=len(loader))
        for idx, (image, target) in enumerate(loader):
            if self.use_cuda:
                image = image.cuda()
                target = target.cuda()

            pred = self.forward(image)
            top1.append(IoU(pred, target))

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

    def extract_feature(self, images):
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
