import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm

from utils import IoU


class KNNSegmentator(nn.Module):

    def __init__(self, backbone, 
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
        self.sum_pool = nn.AvgPool2d(
                            kernel_size=self.patch_size,
                            stride=self.patch_size,
                            divisor_override=1)
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        self.k = k
        self.num_classes = num_classes
        self.feature = feature

        self.use_cuda = use_cuda
        if use_cuda:
            self.backbone.cuda()

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
        for image, target in loader:
            bs = image.size(0)
            
            if self.use_cuda:
                image = image.cuda()
                target = target.cuda()
            
            test_feature = self.extract_feature(image).unsqueeze(3)
            
            # patchwise cosine similarity between test feature & all train features
            # (bs x num_patches x embed_dim x 1) * (num_patches x embed_dim x num_train)
            similarity = (test_feature * self.train_features).sum(dim=2)
            distances, indices = similarity.topk(self.k, largest=True, sorted=True)
            indices = indices.unsqueeze(2).expand(bs, self.num_patches, self.patch_size**2, self.k)
            train_labels = self.train_labels.unsqueeze(0).expand(bs, self.num_patches, self.patch_size**2, -1)
            retrieved_neighbors = torch.gather(train_labels, dim=3, index=indices)

            # tile label patches together
            nrows = self.img_size // self.patch_size
            retrieved_neighbors = retrieved_neighbors.permute(0, 1, 3, 2)
            retrieved_neighbors = retrieved_neighbors.view(bs, self.num_patches, self.k, self.patch_size, self.patch_size)
            retrieved_neighbors = [make_grid(nns, nrows, padding=0) for nns in retrieved_neighbors]
            retrieved_neighbors = torch.stack(retrieved_neighbors)

            top1.append(IoU(retrieved_neighbors[:, 0], target))

            progress_bar.update()
        
        miou = torch.cat(top1, dim=0).mean()

        return miou.item()

    def extract_feature(self, images):
        if self.feature == 'intermediate':
            intermediate_output = self.backbone.get_intermediate_layers(images, self.n_blocks)
            feat = torch.stack(intermediate_output, dim=2)
            feat = torch.mean(feat, dim=2)
            feat = feat[:, 1:]
        else:
            raise NotImplementedError

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
