from datetime import datetime

import numpy as np
import torch
import wandb


CLASS_LABELS_BINARY = {
    0: "background",
    1: "foreground",
    255: "contours"
}

CLASS_LABELS_MULTI = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
    255: "contours"
}
LABELS = [CLASS_LABELS_MULTI[i] for i in range(20)] + ["contours"]

COLOR_MAP_DICT = {
    "background": [83, 135, 222, 153],
    "aeroplane": [218, 77, 77, 153],
    "bicycle": [72, 153, 95, 153],
    "bird": [125, 83, 178, 153],
    "boat": [232, 123, 158, 153],
    "bottle": [227, 117, 55, 153],
    "bus": [135, 207, 192, 153],
    "car": [197, 102, 198, 153],
    "cat": [237, 183, 50, 153],
    "chair": [92, 197, 218, 153],
    "cow": [33, 148, 135, 153],
    "diningtable": [240, 183, 153, 153],
    "dog": [160, 198, 92, 153],
    "horse": [163, 103, 80, 153],
    "motorbike": [162,  40, 100, 153],
    "person": [162, 168, 173, 153],
    "pottedplant": [83, 135, 222, 153],
    "sheep": [218, 77, 77, 153],
    "sofa": [72, 153, 95, 153],
    "train": [125, 83, 178, 153],
    "tvmonitor": [232, 123, 158, 153],
    "contours": [255, 255, 255, 153]
}
COLOR_MAP = np.array([COLOR_MAP_DICT[label] for label in LABELS], dtype=np.uint8)


class WBLogger:

    def __init__(self, args, 
                 project='iBot',
                 entity='dl_lab_enjoyers',
                 group='linear_probe',
                 job_type='vit_base'):
        if args.segmentation == "binary":
            self.class_labels = CLASS_LABELS_BINARY
        else:
            self.class_labels = CLASS_LABELS_MULTI

        self.config = vars(args)

        run_name = datetime.now().strftime('%m.%d-%H:%M') +\
                   f' data={args.percentage} feat={args.feature}'

        wandb.init(
        project=project,
        entity=entity,
        group=group,
        job_type=job_type,
        name=run_name,
        config=self.config)

    def log_segmentation(self, img, pred, segmentation, step, logit=True):
        if logit:
            pred = torch.argmax(pred, dim=1).squeeze(0)
        
        pred_segmentation = wandb.Image(img,
            masks={
                "predictions": {
                    "mask_data": pred.cpu().numpy(),
                    "class_labels": self.class_labels
                    }
                })
        gt_segmentation = wandb.Image(img,
            masks={
                "ground_truth": {
                    "mask_data": segmentation.squeeze(0).cpu().numpy(),
                    "class_labels": self.class_labels
                    }
                })
        wandb.log({
            "Pred Segmentation": pred_segmentation,
            "GT Segmentation": gt_segmentation},
            step=step)

    def log_scalar(self, scalars, step):
        wandb.log(scalars, step=step)

    def log_scalar_summary(self, scalars):
        for key in scalars.keys():
            wandb.run.summary[key] = scalars[key]


def segmentation_to_rgba(mask):
    return COLOR_MAP[mask]
