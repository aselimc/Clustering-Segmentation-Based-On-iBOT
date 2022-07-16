import numpy as np
import torch
import torch.nn.functional as F


def IoU(pred, label, num_classes=21):
    pred = torch.flatten(pred, start_dim=1).long()
    pred = F.one_hot(pred, num_classes)

    label = torch.flatten(label, start_dim=1).long()
    label = F.one_hot(label, num_classes)

    intersection = torch.logical_and(pred, label)
    union = torch.logical_or(pred, label)

    return intersection.sum(dim=[1, 2]) / union.sum(dim=[1, 2])


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
