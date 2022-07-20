from typing import List
from matplotlib.pyplot import grid
from sklearn.cluster import AgglomerativeClustering, Birch

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import argparse
import numpy as np
import models
import tqdm as tqdm

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from dataloader import PartialDatasetVOC
from utils import load_pretrained_weights
import utils.transforms as _transforms
from utils.patch_labeller import LabelPatches
from utils.purity_calculator import best_cluster_count, iteration_over_clusters



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
def Clustering( vit_output: torch.Tensor, n_classes: int, s_label: List):
    # Creating clustering tree
    # bs, z, x, y = vit_output.shape
    vit_output = vit_output[:, 1:, :]
    vit_output = vit_output.reshape( -1, vit_output.shape[2]).contiguous().detach().cpu()
    s_label = s_label.reshape( -1)
    s_label_idx = np.random.randint(low=0, high=len(s_label), size=len(s_label)//10)
    s_label = s_label[s_label_idx]
    labels = np.empty(shape=vit_output.shape[0], dtype="U100")
    labels[s_label_idx] = s_label
    purities = iteration_over_clusters(n_classes,vit_output, labels, s_label_idx)
    
    best_cluster = best_cluster_count(purities)

    cluster = AgglomerativeClustering(n_clusters=best_cluster, compute_distances=True)
    cluster = cluster.fit(vit_output)
    return cluster

def BirchClustering(vit_output, n_clusters):
    vit_output = vit_output[:, 1:, :]
    vit_output = vit_output.reshape( -1, vit_output.shape[2]).contiguous().detach().cpu()

    
    cluster = Birch(n_clusters=n_clusters)

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    val_dataset = datasets.VOCSegmentation(root=args.root, image_set='val', download=False, transforms=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    
    vit_output_list = []
    label_list = []
    for img, label in train_loader:
        vit_output = backbone.get_intermediate_layers(img.cuda(), n=1)[0]
        vit_output_list.append(vit_output.to("cpu"))
        label_list.append(label.to("cpu"))
    vit_output = torch.cat(vit_output_list, dim=0)
    label = torch.cat(label_list, dim=0)

    if args.cluster_algo == "agglo":
        progress_bar = tqdm.tqdm(total=args.n_chunks)
        vit_output = vit_output.chunk(args.n_chunks)
        label = label.chunk(args.n_chunks)
        for idx, item in enumerate(vit_output):
            labels = label[idx]
            n_label, s_label = LabelPatches(labels, patch_size=args.patch_size)
            cluster = Clustering(item, n_classes=21, s_label = s_label)
            plot_dendrogram(cluster, truncate_mode="level", p=3)
            del cluster
            plt.savefig(f"cluster_graphs/cluster_group_{idx}.png")
            plt.clf()
            progress_bar.update()
    else :
        progress_bar = tqdm.tqdm(total=args.n_chunks)
        vit_output = vit_output.chunk(args.n_chunks)
        for idx, item in enumerate(vit_output):
            cluster = BirchClustering(item, n_clusters=100)



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data/VOCtrainval_11-May-2012")
    parser.add_argument('--arch', type=str, default="vit_large")
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--weights', type=str, default="weights/checkpoint.pth")
    parser.add_argument('--segmentation', type=str, choices=['binary', 'multi'], default='multi')
    parser.add_argument('--percentage', type=float, default=1)
    parser.add_argument('--download_data', type=bool, default=False)
    parser.add_argument('--n_chunks', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--cluster_algo', type=str, choices=['agglo', 'birch'], default='agglo')
    return parser.parse_args()




if __name__ == '__main__':
    args = parser_args()
    main(args)
