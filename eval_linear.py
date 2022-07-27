import argparse

from torch.utils.data import DataLoader
from torchvision import datasets

from dataloader import PartialDatasetVOC
from utils.logger import WBLogger
import models
from models.linear import LinearSegmentator
import utils.transforms as _transforms
from utils import load_pretrained_weights


def main(args):
    logger = WBLogger(args, group='linear', job_type=args.arch)

    # Loading the backbone
    backbone = models.__dict__[args.arch](
        patch_size=args.patch_size,
        num_classes=0)
    
    load_pretrained_weights(backbone, args.weights,
                            checkpoint_key="teacher",
                            model_name=args.arch,
                            patch_size=args.patch_size)

    for param in backbone.parameters():
        param.requires_grad = False

    linear_segmentator = LinearSegmentator(backbone,
                                           logger,
                                           num_classes=2 if args.segmentation == 'binary' else 21,
                                           feature=args.feature,
                                           patch_labeling=args.patch_labeling,
                                           background_label_percentage=args.background_label_percentage,
                                           smooth_mask=args.smooth_mask,
                                           n_blocks=args.n_blocks,
                                           epochs=args.epochs,
                                           warmup_epochs=args.warmup_epochs,
                                           lr=args.lr,
                                           eval_freq=args.eval_freq,
                                           use_cuda=True)

    ## TRAINING DATASET ##
    train_transform = _transforms.Compose([
        _transforms.RandomResizedCrop(224),
        _transforms.RandomHorizontalFlip(),
        _transforms.ToTensor(),
        _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ] + ([_transforms.ToBinaryMask()] if args.segmentation == 'binary' else [])
          + [_transforms.MergeContours()]
    )
    val_transform = _transforms.Compose([
        _transforms.Resize(256, interpolation=_transforms.INTERPOLATION_BICUBIC),
        _transforms.CenterCrop(224),
        _transforms.ToTensor(),
        _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ] + ([_transforms.ToBinaryMask()] if args.segmentation == 'binary' else [])
          + [_transforms.MergeContours()]
    )

    train_dataset = PartialDatasetVOC(percentage = args.percentage, root=args.root, image_set='train', download=False, transforms=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)
    val_dataset = datasets.VOCSegmentation(root=args.root, image_set='val', download=False, transforms=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    linear_segmentator.fit(train_loader)
    miou, iou_std = linear_segmentator.score(val_loader)
    print(f'mean intersecion over union: {miou} (±{iou_std}) ')


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data")
    parser.add_argument('--weights', type=str, default="weights/ViT-B.pth")
    parser.add_argument('--arch', type=str, default="vit_base")
    parser.add_argument('--feature', type=str, choices=['intermediate', 'query', 'key', 'value'],
                        default='intermediate')
    parser.add_argument('--patch_labeling', type=str, choices=['coarse', 'fine'], default='coarse')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--smooth_mask', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--percentage", type=float, default=0.1)
    parser.add_argument("--background_label_percentage", type=float, default=0.2)
    parser.add_argument("--segmentation", type=str, choices=['binary', 'multi'], default='multi')
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    main(args)
