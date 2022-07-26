import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm

from utils.logger import CLASS_LABELS_MULTI
import utils.transforms as _transforms


def main(args):
    unfold = nn.Unfold(kernel_size=args.patch_size, stride=args.patch_size)

    ## TRAINING DATASET ##
    transform = _transforms.Compose([
        _transforms.Resize(256, interpolation=_transforms.INTERPOLATION_BICUBIC),
        _transforms.CenterCrop(224),
        _transforms.ToTensor(),
        _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        _transforms.MergeContours()
        ]
    )

    train_dataset = datasets.VOCSegmentation(root=args.root, image_set='train', download=False, transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)

    train_labels = []

    progress_bar = tqdm(total=len(train_loader))
    for _, target in train_loader:
        # divide ground truth mask into patches
        target = unfold(target.unsqueeze(1).float())
        target = target.permute(0, 2, 1).long()
        target = F.one_hot(target, num_classes=21)
        target = torch.argmax(target.sum(dim=2), dim=2)
        target = target.flatten().byte()
        train_labels.append(target)
        progress_bar.update()
    train_labels = torch.cat(train_labels, dim=0).numpy()
    
    classes = list(CLASS_LABELS_MULTI.values())
    classes.remove('contours')
    plt.xticks(np.arange(21), classes, rotation=90)
    plt.hist(train_labels, bins=np.arange(21), align='left', rwidth=0.5)
    plt.savefig(os.path.join('.github', 'histogram.png'), bbox_inches='tight')


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="")
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--percentage', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    main(args)
