import argparse

import os
from xmlrpc.client import boolean
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


from dataloader import PartialDatasetVOC
from utils.logger import WBLogger
import models
from models.classifier import ConvMultiLinearClassifier, ConvSingleLinearClassifier
import utils.transforms as _transforms
from utils import mIoU, MaskedCrossEntropyLoss, load_pretrained_weights
from utils.scheduler import WarmStartCosineAnnealingLR
from clusters import FeatureAgglomerationClustering


global_step = 0


def extract_feature(backbone, img, n_blocks):
    intermediate_output = backbone.get_intermediate_layers(img, n_blocks)
    output = torch.stack(intermediate_output, dim=2)
    output = torch.mean(output, dim=2)
    output = output[:, 1:]
    h, w = int(img.shape[2] / backbone.patch_embed.patch_size), int(img.shape[3] / backbone.patch_embed.patch_size)
    dim = output.shape[-1]
    output = output.reshape(-1, dim, h, w)

    return output


def train(loader, backbone, classifier, logger, criterion, optimizer, n_blocks):
    global global_step
    backbone.eval()
    loss_l = []

    progress_bar = tqdm(total=len(loader))
    for it, (img, segmentation) in enumerate(loader):
        optimizer.zero_grad()
        img = img.cuda()
        segmentation = segmentation.cuda()
        with torch.no_grad():
            output = extract_feature(backbone, img, n_blocks).detach()
        pred_logits = classifier(output, interpolate=False)

        # pred_logits.shape : [BS, 21, 224, 224]
        
        FeatureAgglomerationClustering(pred_logits, 21)
        loss = criterion(pred_logits, segmentation)
        loss.backward()
        optimizer.step()

        loss_l.append(loss.item())
        if it % 10 == 0:
            logger.log_scalar({"training_loss": loss.item()}, step=global_step)

        progress_bar.update()
        global_step += 1

    return np.mean(np.array(loss_l))


def validate(loader, backbone, classifier, logger, criterion, n_blocks):
    backbone.eval()
    classifier.eval()
    val_loss = []
    miou_arr = []
    random_pic_select = np.random.randint(len(loader))
    progress_bar = tqdm(total=len(loader))
    for idx, (img, segmentation) in enumerate(loader):
        img = img.cuda()
        segmentation = segmentation.cuda()
        with torch.no_grad():
            output = extract_feature(backbone, img, n_blocks).detach()
        pred_logits = classifier(output)

        # mask contours: compute pixelwise dummy entropy loss then set it to 0.0
        loss = criterion(pred_logits, segmentation)
        val_loss.append(loss.item())
        miou = mIoU(pred_logits, segmentation)
        miou_arr.append(miou.item())

        if random_pic_select==idx:
            print("Adding Image Example to Logger")
            logger.log_segmentation(img[0], pred_logits, segmentation, step=global_step)
        progress_bar.update()

    return np.mean(np.array(miou_arr)), np.mean(np.array(val_loss))


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

    # Number of blocks of ViT to extract information and feature space dimension based on this
    n_blocks = args.n_blocks
    embed_dim = backbone.embed_dim # * n_blocks

    
    if args.classifier_type == "ConvMultiLinear":
        classifier = ConvMultiLinearClassifier(embed_dim,
                                        n_classes=2 if args.segmentation == 'binary' else 21,
                                        upsample_mode=args.upsample).cuda()
    else:
        classifier = ConvSingleLinearClassifier(embed_dim,
                                                n_classes=2 if args.segmentation == 'binary' else 21,
                                                patch_size=args.patch_size,
                                                upsample_mode=args.upsample).cuda()

    ################################# DATASETS #################################

    # Transformers for training and validation datasets
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

    # Training tools like Optimizer and Scheduler initializations
    if args.optimizer  == "SGD":
        optimizer = optim.SGD(
        classifier.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=0
        )
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(
            classifier.parameters(),
            args.lr,
            betas=(0.9, 0.99),
            weight_decay = 0.01
        )
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    lr_scheduler = WarmStartCosineAnnealingLR(optimizer, args.epochs, args.warmup_epochs, min_lr=0)
    criterion = MaskedCrossEntropyLoss()

    ################################# TRAINING LOOP #################################

    for epoch in range(args.epochs):

        # Validation 
        '''if epoch % args.eval_freq == 0 or epoch == (args.epochs - 1):
            miou, loss = validate(val_loader, backbone, classifier, logger, criterion, n_blocks)
            print(f"Validation for epoch {epoch}: Average mIoU {miou}, Average Loss {loss}")
            logger.log_scalar({
                "val_loss": loss,
                "val_miou": miou,
            }, step=global_step)
'''
        mean_loss = train(train_loader, backbone, classifier, logger, criterion, optimizer, n_blocks)
        print(f"For epoch number {epoch} --> Average Loss {mean_loss:.2f}")
        logger.log_scalar({
            "mean training_loss": mean_loss,
            "learning_rate": float(optimizer.param_groups[0]['lr'])
        }, step=global_step)

        if epoch % args.snapshot_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(args.model_folder, 'ckpt_epoch{}.pth'.format(epoch)))
        lr_scheduler.step()


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data/VOCtrainval_11-May-2012")
    parser.add_argument('--weights', type=str, default="weights/checkpoint.pth")
    parser.add_argument('--arch', type=str, default="vit_large")
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--percentage', type=float, default=1)
    parser.add_argument('--upsample', type=str, choices=['nearest', 'bilinear'], default='bilinear')
    parser.add_argument('--segmentation', type=str, choices=['binary', 'multi'], default='multi')
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--model_folder', type=str, default="classifier_models")
    parser.add_argument('--classifier_type', type=str, choices=['ConvSingleLinear',"ConvMultiLinear"], default="ConvSingleLinear")
    parser.add_argument('--optimizer', type=str, choices=["SGD", "AdamW"], default="SGD")
    parser.add_argument('--snapshot_freq', type=int, default=10)
    parser.add_argument('--download_data', type=bool, default=False)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    main(args)
