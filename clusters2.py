import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from logger import CLASS_LABELS_MULTI
from loader import PartialDatasetVOC
from logger import WBLogger
from operator import itemgetter
import models
from models.classifier import *
import transforms as _transforms
from utils import mIoU, MaskedCrossEntropyLoss, load_pretrained_weights, extract_feature, cosine_scheduler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torchvision.utils import save_image

step = 0

class KMeans_Cluster():
    def __init__(self, n_clusters, backbone, args, logger):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state = 0)
        self.backbone = backbone
        
        self.num_patches = (args.img_size // args.patch_size) * (args.img_size // args.patch_size)
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.num_patches = self.num_patches
        self.n_blocks = args.n_blocks
        self.data_percentage = args.percentage
        self.root = args.root
        self.download_data = args.download_data
        self.batch_size = args.batch_size
        self.logger = logger
        

    def get_data(self, percentage):
        transformations = _transforms.Compose([
            _transforms.Resize(256, interpolation=3),
            _transforms.CenterCrop(224),
            _transforms.ToTensor(),
            _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_dataset = PartialDatasetVOC(percentage = percentage, root=self.root, image_set='train', download=self.download_data, transforms=transformations)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)

        return train_loader

    def save_image_log(self, img, segmentation, prediction):
        """
        prediction -> 196 Tensor 
        """
        global step
        image = torch.zeros(size=(21, 14, 14))
        
        
        for h in range(14):
            for w in range(14):
                image[prediction[ h*w + w ]][h][w] = 1

        image = T.functional.resize(image ,size = (224, 224), interpolation=T.functional.InterpolationMode.NEAREST)
        #save_image(image, "mask.png")
        self.logger.log_segmentation(img, torch.unsqueeze(image, dim=0), segmentation, step)
        step += 1


    def fit_all_data(self, loader):
        """
            Returns clusters
        """
        output, labels = self.get_vit_output(loader)
        
        output = output.reshape(-1, output.shape[2]).contiguous().detach().cpu()
        labels = labels.reshape(-1) #to-do

        clusters = self.kmeans.fit(output)
        return clusters


    def get_vit_output(self, loader):
        """
        Returns output -> # of Data Points x 196 x 1024
        Returns labels -> # of Data Points x 224 x 224

        """
        all_output = []
        labels = []
        for img, label in loader:
            output = self.backbone.get_intermediate_layers(img.cuda(), n=1)[0]
            all_output.append(output.cpu())
            labels.append(label.cpu())
        
        output = torch.cat(all_output, dim=0)
        labels = torch.cat(labels, dim=0)

        return output[:, 1:, :], labels #discard CLS token

    def predict_for_image(self, img, segmentation, vit_output_single_img, clusters, assigned_labels_for_clusters, save_image = True):
        """
            cluster_centroids -> K x 1024  Tensor
            assigned_labels_for_clusters -> K x 21 Tensor (one hot encoded)
            vit_output_single_img -> 196 x 1024

            returns prediction -> 196
        """
        
        prediction = clusters.predict(vit_output_single_img)
            # prediction -> 196 Tensor with cluster centroid ids

        for patch in range(prediction.shape[0]):
            prediction[patch] = torch.argmax(assigned_labels_for_clusters[prediction[patch]])
            # prediction -> 196 with class ids (0 1 2...21)

        if save_image:
            self.save_image_log(img, segmentation, prediction)

        return prediction

    def label_patches(self, segmentation):
        """
            segmentation -> b x h x w (32, 224, 224)

            Returns:
            patch_labels -> 32 x 14 x 14
            label_names -> 32 x 14 x 14
        """
        b, h, w = segmentation.shape 
        patch_labels = torch.empty(size=(b, h//self.patch_size, w//self.patch_size))
        label_names = np.empty(shape=(b, h//self.patch_size, w//self.patch_size), dtype='U100')

        for rows in range(h//self.patch_size):
            for columns in range(w//self.patch_size):
                vote = self.majority_voter(segmentation[:, self.patch_size*rows : self.patch_size*(rows+1), self.patch_size*columns : self.patch_size*(columns+1)])
                patch_labels[:, rows, columns] = vote
                x = vote.int().tolist()
                label_names[:, rows, columns] = np.array(itemgetter(*x)(CLASS_LABELS_MULTI))
        
        return patch_labels, label_names

    def majority_voter(self, slice: torch.Tensor):
        slice = slice.reshape(slice.shape[0], -1).contiguous()
        maj = []
        for i, rows in enumerate(slice):
            idx = torch.argmax(torch.unique(rows, return_counts=True)[1]).item()
            idx = torch.unique(rows, return_counts=True)[0][idx]
            maj.append(idx)
        return torch.Tensor(maj)

    def label_clusters(self, clusters, vit_o_ld, patch_labels_ld, label_names_ld):
        """
        clusters: including cluster centers -> K x 1024
        vit_o_ld: vit output of labelled data -> # of data points x 14 x 14
        patch_labels_ld: labels for the patches -> # of data points x 14 x 14
        label_names_ld: label names for patches -> # of data points x 14 x 14
        """

        assigned_labels_for_clusters = torch.zeros(size=(clusters.cluster_centers_.shape[0], 21))
        actual_labels_for_clusters = torch.zeros(size=(clusters.cluster_centers_.shape[0], 21))
        patch_labels_ld = torch.flatten(patch_labels_ld, start_dim=1)
        #label_names_ld = torch.flatten(label_names_ld, start_dim=1)

        for img_idx in range(vit_o_ld.size(0)):
            prediction = clusters.predict(vit_o_ld[img_idx])
            # prediction -> 196 Tensor with cluster centroid ids

            for patch_idx in range(prediction.size):
                idx1 = prediction[patch_idx]
                idx2 = int(patch_labels_ld[img_idx][patch_idx].item())
                if idx2 == 255:
                    idx2 = 0
                assigned_labels_for_clusters[idx1][idx2] += 1
        
        for c in range(assigned_labels_for_clusters.size(0)):
            topk = torch.topk(assigned_labels_for_clusters[c], 2)
            
            if(topk.values[1] * 10 > topk.values[0]):
                actual_labels_for_clusters[c][topk.indices[1]] = 1
            else:
                actual_labels_for_clusters[c][topk.indices[0]] = 1
             
        return actual_labels_for_clusters

    def run_kmeans(self):
        # Initialize data loaders
        whole_data = self.get_data(0.1) # change it to 1. !!
        labelled_data = self.get_data(self.data_percentage)

        # Fit whole data and form clusters
        clusters = self.fit_all_data(whole_data)

        # Get the vit output for labelled data (ld)
        vit_o_ld, labels_ld= self.get_vit_output(labelled_data)
        patch_labels_ld, label_names_ld = self.label_patches(labels_ld)

        # Label the cluster centroids
        assigned_labels_for_clusters = self.label_clusters(clusters, vit_o_ld, patch_labels_ld, label_names_ld)
        #assigned_labels_for_clusters = self.majority_labeller(clusters, 20, patch_labels_ld, label_names_ld)
        
        # Get the vit output for test images
        vit_o_test, _ = self.get_vit_output(whole_data) 

        # Make the predictions for test images
        for img, segmentation in whole_data:
            save_image(img[0], "img0.png")
            self.predict_for_image(img[0], segmentation[0], vit_o_test[0], clusters, assigned_labels_for_clusters, save_image = True)
            save_image(img[1], "img1.png")
            self.predict_for_image(img[1], segmentation[1], vit_o_test[1], clusters, assigned_labels_for_clusters, save_image = True)
            save_image(img[2], "img2.png")
            self.predict_for_image(img[2], segmentation[2], vit_o_test[2], clusters, assigned_labels_for_clusters, save_image = True)
            save_image(img[3], "img3.png")
            self.predict_for_image(img[3], segmentation[3], vit_o_test[3], clusters, assigned_labels_for_clusters, save_image = True)
            save_image(img[4], "img4.png")
            self.predict_for_image(img[4], segmentation[4], vit_o_test[4], clusters, assigned_labels_for_clusters, save_image = True)



        return 



def main(args):
    logger = WBLogger(args)
    ################################# BACKBONE & CLASSIFIER INIT #################################
    # Loading backbone ViT from the models dir

    backbone = models.__dict__[args.arch](
        patch_size=args.patch_size,
        num_classes=0
    )
    backbone = backbone.cuda()

    # Loading the weights from a full .ckpt file that can be downloaded from 
    # https://github.com/aselimc/iBot-cv/blob/main/README.md#pre-trained-models

    load_pretrained_weights(backbone, args.weights, "teacher", args.arch, args.patch_size)

    # Freezing the backbone weights
    for param in backbone.parameters():
        param.requires_grad = False

    kmeans_method = KMeans_Cluster(n_clusters = 80, backbone = backbone, args = args, logger = logger)

    kmeans_method.run_kmeans()
    
    
    
    #kmeans_method.fit_labelled_data()


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data/VOCtrainval_11-May-2012")
    parser.add_argument('--arch', type=str, default="vit_large")
    parser.add_argument('--weights', type=str, default="/project/dl2022s/oezoegeb/iBot-cv/weights/checkpoint_teacher_vitL.pth")
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--percentage', type=float, default=0.1)
    parser.add_argument('--download_data', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--segmentation', type=str, choices=['binary', 'multi'], default='multi')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--in_chans', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=768)

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    main(args)