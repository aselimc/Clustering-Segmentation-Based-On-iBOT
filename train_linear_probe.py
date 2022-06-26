import argparse
import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transforms import *
from torch.utils.data import DataLoader
from PIL import Image
import models


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="data")
    parser.add_argument('--weights', default="weights/ViT-S16.pth")
    parser.add_argument('--arch', default="vit_small")
    parser.add_argument('--patch_size', default=16)

    return parser.parse_args()

def train(loader, model):
    
    for img, segmentation in loader:
        out = model(img)
        out = model(out)
        out = model(out)

        print(out.shape)

def main(args):

    # Loading the backbone
    backbone = models.__dict__[args.arch](
        patch_size=args.patch_size,
        return_all_tokens=True,
    )

    state_dict = torch.load(args.weights)['state_dict']
    backbone.load_state_dict(state_dict)
    
    for param in backbone.parameters():
    param.requires_grad = False

    train_transform = Compose([
        RandomCrop(224),
        PILToTensor(),
        MergeContours(),
        RandomHorizontalFlip(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    dataset = datasets.VOCSegmentation(root=args.root, image_set='train', download=False, transforms=train_transform)
    loader = DataLoader(dataset, batch_size=16)
    flatten = nn.Flatten()
    linear = nn.Linear(75648,20 )
    model = nn.Sequential(backbone, flatten, linear)
    for epoch in args.epochs:
        train(loader, model)


def show_img(img, segmentation):
    img_pil = transforms.functional.to_pil_image(img)
    seg_pil = transforms.functional.to_pil_image(segmentation)
    img_pil.show()
    seg_pil.show()

if __name__ == '__main__':
    args = parser_args()
    main(args)
