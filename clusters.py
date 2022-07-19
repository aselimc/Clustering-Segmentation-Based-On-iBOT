from matplotlib.pyplot import grid
from sklearn.cluster import AgglomerativeClustering

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import argparse
import numpy as np
import models

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from dataloader import PartialDatasetVOC
from utils import load_pretrained_weights
import utils.transforms as _transforms



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# We fit data, we fit labeled data, and we label clusters
def Clustering( vit_output: torch.Tensor, n_clusters: int):
    # Creating clustering tree
    # bs, z, x, y = vit_output.shape
    vit_output = vit_output[:, 1:, :]
    vit_output = vit_output.reshape( -1, vit_output.shape[2]).contiguous().detach().cpu()

    
    cluster = AgglomerativeClustering(n_clusters=n_clusters, compute_distances=True)
    cluster = cluster.fit(vit_output)
    return cluster


def main(args):
    backbone = models.__dict__[args.arch](
        patch_size=args.patch_size,
        num_classes=0
    )
    backbone = backbone.cuda()
    load_pretrained_weights(backbone, args.weights, "teacher", args.arch, args.patch_size)
    transformations = [
        _transforms.RandomResizedCrop(224),
        _transforms.RandomHorizontalFlip(),
        _transforms.ToTensor(),
        _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    val_transformations = [
        _transforms.Resize(256, interpolation=_transforms.INTERPOLATION_BICUBIC),
        _transforms.CenterCrop(224),
        _transforms.ToTensor(),
        _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    if args.segmentation == 'binary':
        transformations.append(_transforms.ToBinaryMask())
        val_transformations.append(_transforms.ToBinaryMask())
    else:
        transformations.append(_transforms.MergeContours())
        val_transformations.append(_transforms.MergeContours())

    train_transform = _transforms.Compose(transformations)
    val_transform = _transforms.Compose(val_transformations)

    for param in backbone.parameters():
        param.requires_grad = False
    # Dataset and Loader initializations
    train_dataset = PartialDatasetVOC(percentage = args.percentage, root=args.root, image_set='train', download=args.download_data, transforms=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4)
    val_dataset = datasets.VOCSegmentation(root=args.root, image_set='val', download=False, transforms=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    
    vit_output_list = []
    for img, label in train_loader:
        vit_output = backbone.get_intermediate_layers(img.cuda(), n=1)[0]
        vit_output_list.append(vit_output.to("cpu"))
    vit_output = torch.cat(vit_output_list, dim=0)

    vit_output = vit_output.chunk(20)
    for idx, item in enumerate(vit_output):
        cluster = Clustering(item, n_clusters=100)
        plot_dendrogram(cluster, truncate_mode="level", p=3)
        del cluster
        plt.savefig(f"cluster_group_{idx}.png")



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data/VOCtrainval_11-May-2012")
    parser.add_argument('--arch', type=str, default="vit_large")
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--weights', type=str, default="weights/checkpoint.pth")
    parser.add_argument('--segmentation', type=str, choices=['binary', 'multi'], default='multi')
    parser.add_argument('--percentage', type=float, default=1)
    parser.add_argument('--download_data', type=bool, default=False)
    return parser.parse_args()




if __name__ == '__main__':
    args = parser_args()
    main(args)
