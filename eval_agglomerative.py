import argparse

from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from dataloader import PartialDatasetVOC
import models
from models.agglomerative import AgglomerativeClustering
from utils.logger import WBLogger
import utils.transforms as _transforms
from utils import load_pretrained_weights


def main(args):
    logger = WBLogger(args, group='agglomerative', job_type=args.arch)

    # Loading the backbone
    backbone = models.__dict__[args.arch](
        patch_size=args.patch_size,
        num_classes=0
    )

    load_pretrained_weights(backbone, args.weights,
                            checkpoint_key="teacher",
                            model_name=args.arch,
                            patch_size=args.patch_size)

    agglomerative = AgglomerativeClustering(
                    backbone,
                    logger,
                    n_clusters=args.n_clusters,
                    n_chunks=args.n_chunks,
                    feature=args.feature,
                    n_blocks=args.n_blocks,
                    use_cuda=True,
                    distance=args.distance,
                    calculate_purity=args.purity,
                    patch_labeling=args.patch_labeling,
                    percentage=args.label_percentage)

    ## TRAINING DATASET ##
    transform = _transforms.Compose([
        _transforms.Resize(256, interpolation=_transforms.INTERPOLATION_BICUBIC),
        _transforms.CenterCrop(224),
        _transforms.ToTensor(),
        _transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ] + ([_transforms.ToBinaryMask()] if args.segmentation == 'binary' else [])
          + [_transforms.MergeContours()]
    )

    train_dataset = PartialDatasetVOC(percentage = args.percentage, root=args.root, image_set='train', download=False, transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers)
    val_dataset = datasets.VOCSegmentation(root=args.root, image_set='val', download=False, transforms=transform)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # agglomerative.fit(train_loader)
    agglomerative.load_cluster_centroids()
    agglomerative.forward(val_loader)
    # miou, iou_std = kmeans.score(val_loader)
    # print(f'mean intersecion over union: {miou} (±{iou_std}) ')


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data/VOCtrainval_11-May-2012")
    parser.add_argument('--weights', type=str, default="weights/checkpoint.pth")
    parser.add_argument('--arch', type=str, default="vit_large")
    parser.add_argument('--feature', type=str, choices=['intermediate', 'query', 'key', 'value'],
                        default='intermediate')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--n_blocks', type=int, default=1)
    parser.add_argument('--patch_labeling', type=str, choices=['coarse', 'fine'], default='fine')
    parser.add_argument('--n_neighbors', type=int, default=20)
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--tol', type=float, default=1e-4),
    parser.add_argument('--distance', type=str, choices=['euclidean', 'cosine'], default='euclidean')  
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--percentage", type=float, default=1)
    parser.add_argument("--label_percentage", type=float, default=1)
    parser.add_argument("--segmentation", type=str, choices=['binary', 'multi'], default='multi')
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--n_chunks', type=int, default=20)
    parser.add_argument('--purity', type=bool, default=False)
    parser.add_argument('--n_clusters', type=int, default=80)


    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    main(args)