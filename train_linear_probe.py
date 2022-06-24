import argparse
import torch
import torch.nn as nn

import torchvision.datasets as datasets
from transforms import *
from torch.utils.data import DataLoader

import models


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--weights')
    parser.add_argument('--arch', default="vit_small")
    parser.add_argument('--patch_size', default=16)

    return parser.parse_args()

def train(args):
    backbone = models.__dict__[args.arch](
        patch_size=args.patch_size,
        return_all_tokens=True,
    )

    state_dict = torch.load(args.weights)['state_dict']
    backbone.load_state_dict(state_dict)
    
    train_transform = Compose([
        RandomCrop(224),
        PILToTensor(),
        MergeContours(),
        RandomHorizontalFlip(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = datasets.VOCSegmentation(root=args.root, image_set='val', download=False, transforms=train_transform)
    loader = DataLoader(dataset, batch_size=16)

    for img, segmentation in loader:
        out = backbone(img)

        print(out.shape)


if __name__ == '__main__':
    args = parser_args()
    train(args)
