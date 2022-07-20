import numpy as np
import torch
from utils.logger import CLASS_LABELS_MULTI
from operator import itemgetter

def majority_voter(slice: torch.Tensor):
    slice =     slice.reshape(slice.shape[0], -1).contiguous()
    maj = []
    for i, rows in enumerate(slice):
        idx = torch.argmax(torch.unique(rows, return_counts=True)[1]).item()
        idx = torch.unique(rows, return_counts=True)[0][idx]
        maj.append(idx)
    return torch.Tensor(maj)

def LabelPatches(seg, patch_size):
    bs, h, w = seg.shape
    patch_labels = torch.empty(size=(bs, h//patch_size, w//patch_size))
    label_names = np.empty(shape=(bs, h//patch_size, w//patch_size), dtype='U100')
    for rows in range(h//patch_size):
        for columns in range(w//patch_size):
            vote = majority_voter(seg[:, patch_size*rows : patch_size*(rows+1), patch_size*columns : patch_size*(columns+1)])
            patch_labels[:, rows, columns] = vote
            x = vote.int().tolist()
            label_names[:, rows, columns] = np.array(itemgetter(*x)(CLASS_LABELS_MULTI))
    return patch_labels, label_names
    

if __name__ == '__main__':
    x = torch.randint(255, size=(64, 21, 224, 224))
    y = torch.randint(20, size = (64, 224, 224))
    patch_labels = LabelPatches(x, y, 16)
    x = 5