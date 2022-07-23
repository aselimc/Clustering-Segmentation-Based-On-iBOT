from typing import List
from sklearn.cluster import AgglomerativeClustering

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import argparse
import numpy as np
import models
from tqdm import tqdm

from scipy.cluster.hierarchy import dendrogram
from dataloader import PartialDatasetVOC
from utils import load_pretrained_weights
from utils.logger import WBLogger
from utils.metrics import mIoU
import utils.transforms as _transforms
from utils.patch_labeller import LabelPatches
from utils.purity_calculator import *

global global_step
global_step = 0


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
def Clustering( vit_output: torch.Tensor, real_labels: List, start, stop, step, logger):
    # Creating clustering tree

    vit_output = vit_output[:, 1:, :]
    vit_output = vit_output.reshape( -1, vit_output.shape[2]).contiguous().detach().cpu()
    real_labels = real_labels.reshape( -1)
    # s_label_idx = equal_random_selector(real_labels)
    s_label_idx = np.arange(len(real_labels))
    s_label = real_labels[s_label_idx]
    labels = np.empty(shape=vit_output.shape[0], dtype="U100")
    labels[s_label_idx] = s_label
    # print("Trying different number of clusters to find optimal one")
    #purities = iteration_over_clusters(vit_output, labels, s_label_idx, start, stop, step, logger)
    
    # best_cluster = best_cluster_count(purities, start, step)
    # print(f"Best number of cluster that has been found is {best_cluster}")
    best_cluster = 80
    cluster = AgglomerativeClustering(n_clusters=best_cluster, affinity="euclidean",
                                      compute_full_tree=False, linkage="ward", compute_distances=True)
    cluster = cluster.fit(vit_output)
    labels_entire_trainset = majority_labeller(cluster, best_cluster, s_label_idx, labels)
    class_means = get_class_means(vit_output, labels_entire_trainset)
    return class_means

def validate(loader, class_means, backbone, logger):
    global global_step
    backbone.eval()
    miou_arr = []
    progress_bar = tqdm(total=len(loader))

    for idx, (img, segmentation) in enumerate(loader):
        global_step += 1
        img = img.cuda()
        with torch.no_grad():
            vit_output = backbone.get_intermediate_layers(img.cuda(), n=1)[0]
            vit_output = vit_output[:, 1:, :]
            vit_output = vit_output.reshape( -1, vit_output.shape[2]).contiguous().detach().cpu()
            image_labels = predict(class_means, vit_output)
            miou = mIoU(image_labels, segmentation, no_softmax=False)
            miou_arr.append(miou)
            progress_bar.update()
            if idx % 5 ==0:
                logger.log_cluster_segmentation(img[0], image_labels, segmentation, global_step)
    mean = np.mean(np.array(miou_arr))
    print(f"For validation set the average miou is {mean:.2f}")
    return mean
            

def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

def main(args):
    global global_step
    logger = WBLogger(args)


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

    vit_output = vit_output.chunk(args.n_chunks)
    label = label.chunk(args.n_chunks)
    
    progressbar = tqdm(total=args.n_chunks)
    c_means=[]
    for item, labels in zip(vit_output, label):
        global_step += 1
        _, s_label = LabelPatches(labels, patch_size=args.patch_size)

        class_means=Clustering(item,
                            real_labels = s_label,
                            start = args.n_cluster_start, 
                            stop = args.n_cluster_stop, 
                            step = args.n_cluster_step,
                            logger= logger)
        miou = validate(val_loader, class_means, backbone, logger)
        logger.log_scalar({"val_miou": miou,
            }, step=global_step)
        c_means.append(class_means)
        progressbar.update()

    class_means = dict_mean(c_means)
    miou = validate(val_loader, class_means, backbone, logger)

        




def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data/VOCtrainval_11-May-2012")
    parser.add_argument('--arch', type=str, default="vit_large")
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--weights', type=str, default="weights/checkpoint.pth")
    parser.add_argument('--segmentation', type=str, choices=['binary', 'multi'], default='multi')
    parser.add_argument('--percentage', type=float, default=1)
    parser.add_argument('--download_data', type=bool, default=False)
    parser.add_argument('--n_chunks', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--n_cluster_start', type=int, default=96)
    parser.add_argument('--n_cluster_stop', type=int, default=100)
    parser.add_argument('--n_cluster_step', type=int, default=4)
    return parser.parse_args()




if __name__ == '__main__':
    args = parser_args()
    main(args)
