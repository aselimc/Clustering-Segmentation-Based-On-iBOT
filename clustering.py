import models
import torch
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from train_linear_probe import  parser_args, train
import transforms as _transforms
from utils import load_pretrained_weights, extract_feature
from loader import PartialDatasetVOC
from torchvision import datasets
from torch.utils.data import DataLoader
from models.classifier import *

def main(args):
    n_classes = 21 if args.segmentation == "multi" else 2
    clustering = FeatureAgglomeration(n_clusters=n_classes)
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

    # Number of blocks of ViT to extract information and feature space dimension based on this
    n_blocks = args.n_blocks
    embed_dim = backbone.embed_dim # * n_blocks
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

    # Dataset and Loader initializations
    train_dataset = PartialDatasetVOC(percentage = args.percentage, root=args.root, image_set='train', download=args.download_data, transforms=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataset = datasets.VOCSegmentation(root=args.root, image_set='val', download=False, transforms=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1)
    classifier = ConvSingleLinearClassifier(embed_dim,
                                      n_classes=2 if args.segmentation == 'binary' else 21,
                                      upsample_mode=args.upsample, patch_size=args.patch_size).cuda()

    for img,seg in train_loader:
        out = extract_feature(backbone, img, 1)
        out = classifier(out, upsample_flag=False)
        out = out.view(out.shape[0], -1)
        out = clustering.fit_transform(out.detach().cpu())
        out = clustering.inverse_transform(out)
        


if __name__ == "__main__":
    args = parser_args()
    main(args)