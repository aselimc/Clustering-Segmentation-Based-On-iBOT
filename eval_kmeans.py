import argparse

from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from utils.dataloader import PartialDatasetVOC
import models
from models.kmeans import KMeansSegmentator
from utils.logger import WBLogger
import utils.transforms as _transforms
from utils import load_pretrained_weights


def main(args):
    logger = WBLogger(args, group='kmeans', job_type=args.arch)

    # Loading the backbone
    backbone = models.__dict__[args.arch](
        patch_size=args.patch_size,
        num_classes=0
    )

    load_pretrained_weights(backbone, args.weights,
                            checkpoint_key="teacher",
                            model_name=args.arch,
                            patch_size=args.patch_size)

    kmeans = KMeansSegmentator(backbone,
                    logger,
                    k=args.n_centroids,
                    feature=args.feature,
                    smooth_mask=args.smooth_mask,
                    n_blocks=args.n_blocks,
                    use_cuda=True,
                    distance=args.distance,
                    init_mode=args.init,
                    n_redo=args.n_init,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    percentage=args.percentage,
                    background_label_percentage=args.background_label_percentage)

    ## TRAINING DATASET ##
    transform = _transforms.Compose([
        _transforms.Resize(256, interpolation=_transforms.INTERPOLATION_BICUBIC),
        _transforms.CenterCrop(224),
        _transforms.ToTensor(),
        _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ] + ([_transforms.ToBinaryMask()] if args.segmentation == 'binary' else [])
          + [_transforms.MergeContours()]
    )

    train_dataset = PartialDatasetVOC(percentage=1.0, root=args.root, image_set='train', download=False, transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)
    val_dataset = datasets.VOCSegmentation(root=args.root, image_set='val', download=False, transforms=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    kmeans.fit(train_loader)
    miou, iou_std = kmeans.score(val_loader)
    print(f'mean intersecion over union: {miou} (±{iou_std}) ')


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data")
    parser.add_argument('--weights', type=str, default="weights/ViT-S16.pth")
    parser.add_argument('--arch', type=str, default="vit_base")
    parser.add_argument('--feature', type=str, choices=['intermediate', 'query', 'key', 'value'],
                        default='intermediate')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--n_blocks', type=int, default=1)
    parser.add_argument('--patch_labeling', type=str, choices=['coarse', 'fine'], default='fine')
    parser.add_argument('--n_centroids', type=int, default=20)
    parser.add_argument('--smooth_mask', action='store_true')
    parser.add_argument('--weighted_majority_vote', action='store_true')
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--tol', type=float, default=1e-4),
    parser.add_argument('--init', type=str, choices=['kmeans++', 'random'], default='kmeans++')
    parser.add_argument('--n_init', type=int, default=10)
    parser.add_argument('--distance', type=str, choices=['euclidean', 'cosine'], default='euclidean')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--percentage", type=float, default=0.1)
    parser.add_argument("--background_label_percentage", type=float, default=1.0)
    parser.add_argument("--segmentation", type=str, choices=['binary', 'multi'], default='multi')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    main(args)
