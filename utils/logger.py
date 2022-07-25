from datetime import datetime
from this import d

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


class WBLogger:

    def __init__(self, args, 
                 project="iBot",
                 entity="dl_lab_enjoyers",
                 group='linear_probe',
                 job_type='vit_base'):
        if args.segmentation == "binary":
            self.class_labels = CLASS_LABELS_BINARY
        else:
            self.class_labels = CLASS_LABELS_MULTI

        self.config = vars(args)

        wandb.init(
        project=project,
        entity=entity,
        group=group,
        job_type=job_type,
        name=datetime.now().strftime('%m.%d.%Y-%H:%M:%S'),
        config=self.config)
        
    def log_segmentation(self, img, pred_logits, segmentation, step):
        pred_segmentation = wandb.Image(img,
            masks={
                "predictions": {
                    "mask_data": torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy(),
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

    def log_cluster_segmentation(self, img, preds, segmentation, step):
        pred_segmentation = wandb.Image(img,
            masks={
                "predictions": {
                    "mask_data": preds.squeeze().squeeze().cpu().numpy(),
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
    
