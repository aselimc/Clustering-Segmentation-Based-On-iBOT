import numpy as np
import torch
import torch.nn.functional as F


def mIoU(pred, gt, num_classes=21):
    ins_iou = []
    for instance in range(num_classes):
        if instance==0:
            continue #do not consider background
        intersection = ((pred == instance) & (gt == instance)).sum().float()
        union = ((pred == instance) | (gt == instance)).sum().float()
        if union==0:
            continue
        iou_val = intersection/(union+1.)
        ins_iou.append(iou_val)

    mean_iou = torch.mean(torch.stack(ins_iou))
    return mean_iou


def mIoUWithLogits(pred, label, num_classes=21):
    pred = torch.softmax(pred.float(), dim=1)              
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)
