import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .vision_transformer import VisionTransformer, vit_tiny, vit_small, vit_base, vit_large
from utils.metrics import mIoU
from utils.transforms import PatchwiseSmoothMask


class _BaseSegmentator(nn.Module):

    def __init__(self, backbone,
                 logger,
                 num_classes=21,
                 feature='intermediate',
                 patch_labeling='coarse',
                 background_label_percentage=0.2,
                 smooth_mask=True,
                 n_blocks=1,
                 use_cuda=True):
        """
        Args:
            feature: 'intermediate', 'query', 'key', 'value'
            patch_label: 'coarse', 'fine'
        """
        super(_BaseSegmentator, self).__init__()

        self.backbone = backbone
        self.n_blocks = n_blocks
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        self.logger = logger

        self.num_classes = num_classes
        self.feature = feature
        self.patch_labeling = patch_labeling
        self.background_label_percentage = background_label_percentage
        self.smooth_mask = smooth_mask

        if smooth_mask:
            self.smooth = PatchwiseSmoothMask(self.patch_size)
        else:
            self.smooth = nn.Identity()

        self.use_cuda = use_cuda
        if use_cuda:
            self.backbone.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, image):
        raise NotImplementedError

    def fit(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def score(self, loader):
        top1 = []
        
        print("Compute score...")
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
                self.logger.log_segmentation(image[2], pred[2], target[2], step=idx+1, logit=False)
            progress_bar.update()

        top1 = torch.stack(top1, dim=0)
        miou = torch.mean(top1).item()
        iou_std = torch.std(top1).item()

        self.logger.log_scalar_summary({
                "mIoU": miou,
                "IoU std": iou_std,
            })

        return miou, iou_std

    @torch.no_grad()
    def _transform_data(self, loader):
        train_features = []
        train_labels = []
        self.cluster_labels = []

        print("Extracting ViT features...")
        progress_bar = tqdm(total=len(loader))
        for image, target in loader:
            image = image.to(device=self.device)
            feat = self._extract_feature(image)
            feat = feat.cpu()
            train_features.append(feat)

            target = target.to(device=self.device)
            target = self._mask_to_patches(target)
            target = target.cpu()
            train_labels.append(target)
            progress_bar.update()

        train_features = torch.cat(train_features, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        return train_features, train_labels

    @torch.no_grad()
    def _extract_feature(self, image, flatten=True):
        if self.feature == 'intermediate':
            intermediate_output = self.backbone.get_intermediate_layers(image, self.n_blocks)
        else:
            intermediate_output = self.backbone.get_qkv(image, self.n_blocks, out=self.feature)
        
        feat = torch.stack(intermediate_output, dim=2)
        feat = torch.mean(feat, dim=2)
        feat = feat[:, 1:]

        if flatten:
            feat = feat.flatten(start_dim=0, end_dim=1)

        return feat

    @torch.no_grad()
    def _mask_to_patches(self, mask):
        # divide ground truth mask into patches
        mask = self.unfold(mask.unsqueeze(1).float())
        mask = mask.permute(0, 2, 1)
        if self.patch_labeling == 'coarse':
            mask = mask.long()
            mask = F.one_hot(mask, self.num_classes)
            mask = torch.argmax(mask.sum(dim=2), dim=2)
            mask = mask.unsqueeze(2).expand(-1, self.num_patches, self.patch_size**2)
        mask = mask.flatten(start_dim=0, end_dim=1)
        mask = mask.byte()

        return mask

    @torch.no_grad()
    def _balance_class(self, train_features, train_labels):
        is_background = (train_labels == 0).all(dim=1)
        idx_background = torch.nonzero(is_background, as_tuple=False).squeeze(1).numpy()
        idx_foreground = torch.nonzero(~is_background, as_tuple=False).squeeze(1).numpy()
        size_undersample = int(len(idx_background) * self.background_label_percentage)
        idx_undersample = np.random.choice(size_undersample, size=size_undersample, replace=False)
        idx_train = np.concatenate([idx_foreground, idx_undersample], axis=0)

        train_features = train_features[idx_train]
        train_labels = train_labels[idx_train]

        return train_features, train_labels

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