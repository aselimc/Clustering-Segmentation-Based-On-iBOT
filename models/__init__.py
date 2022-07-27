import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .vision_transformer import VisionTransformer, vit_tiny, vit_small, vit_base, vit_large
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

    def score(self, loader):
        raise NotImplementedError

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
