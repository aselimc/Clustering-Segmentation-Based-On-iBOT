import argparse
from datetime import datetime

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
from models.classifier import UNet
import utils.transforms as _transforms
from utils import mIoUWithLogits, MaskedCrossEntropyLoss


global_step = 0


def train(loader, classifier, logger, criterion, optimizer):
    global global_step

    classifier.train()
    loss_l = []

    progress_bar = tqdm(total=len(loader))
    for it, (img, segmentation) in enumerate(loader):
        optimizer.zero_grad()
        img = img.cuda()
        segmentation = segmentation.cuda()
        pred_logits = classifier(img)

        loss = criterion(pred_logits, segmentation)
        loss.backward()
        optimizer.step()

        loss_l.append(loss.item())
        if it % 10 == 0:
            logger.log_scalar({"training_loss": loss.item()}, step=global_step)

        progress_bar.update()
        global_step += 1

    return np.mean(np.array(loss_l))


def validate(loader, classifier, logger, criterion):
    classifier.eval()
    val_loss = []
    miou_arr = []
    random_pic_select = np.random.randint(len(loader))
    for idx, (img, segmentation) in enumerate(loader):
        img = img.cuda()
        segmentation = segmentation.cuda()
        pred_logits = classifier(img)

        # mask contours: compute pixelwise dummy entropy loss then set it to 0.0
        loss = criterion(pred_logits, segmentation)
        val_loss.append(loss.item())
        miou = mIoUWithLogits(pred_logits, segmentation)
        miou_arr.append(miou.item())

        if random_pic_select==idx:
            print("Adding Image Example to Logger")
            logger.log_segmentation(img[0], pred_logits, segmentation, step=global_step)

    return np.mean(np.array(miou_arr)), np.mean(np.array(val_loss))


def main(args):
    logger = WBLogger(args)

    classifier = UNet(n_channels=3,
                      n_classes=2 if args.segmentation == 'binary' else 21,
                      bilinear=(args.upsample == 'bilinear')).cuda()

    ## TRAINING DATASET ##
    train_transform = _transforms.Compose([
        _transforms.RandomResizedCrop(224),
        _transforms.RandomHorizontalFlip(),
        _transforms.ToTensor(),
        _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ] + [_transforms.ToBinaryMask()] if args.segmentation == 'binary' else []
    )
    val_transform = _transforms.Compose([
        _transforms.Resize(256, interpolation=_transforms.INTERPOLATION_BICUBIC),
        _transforms.CenterCrop(224),
        _transforms.ToTensor(),
        _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ] + [_transforms.ToBinaryMask()] if args.segmentation == 'binary' else []
    )

    train_dataset = PartialDatasetVOC(percentage = args.percentage, root=args.root, image_set='train', download=False, transforms=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataset = datasets.VOCSegmentation(root=args.root, image_set='val', download=False, transforms=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1)

    optimizer = optim.SGD(
        classifier.parameters(),
        args.lr * args.batch_size / 256.0,
        momentum=0.9,
        weight_decay=0
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    criterion = MaskedCrossEntropyLoss()

    ############## TRAINING LOOP #######################

    for epoch in range(args.epochs):
        mean_loss = train(train_loader, classifier, logger, criterion, optimizer)
        print(f"For epoch number {epoch} --> Average Loss {mean_loss:.2f}")
        logger.log_scalar({
            "mean training_loss": mean_loss,
            "learning_rate": lr_scheduler.get_last_lr()[0]
        }, step=global_step)
        if epoch % args.eval_freq == 0 or epoch == (args.epochs - 1):
            miou, loss = validate(val_loader, classifier, logger, criterion)
            print(f"Validation for epoch {epoch}: Average mIoU {miou}, Average Loss {loss}")
            logger.log_scalar({
                "val_loss": loss,
                "val_miou": miou,
            }, step=global_step)

        lr_scheduler.step()


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--percentage", type=float, default=0.1)
    parser.add_argument("--upsample", type=str, choices=['nearest', 'bilinear'], default='nearest')
    parser.add_argument("--segmentation", type=str, choices=['binary', 'multi'], default='multi')
    parser.add_argument("--eval_freq", type=int, default=5)

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    main(args)