import torch.nn as nn


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, mask_val=255):
        super(MaskedCrossEntropyLoss, self).__init__()
        
        self.mask_val = mask_val
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, input, target):
        mask = (target == self.mask_val)
        num_mask_el = mask.sum()
        target[mask] = 0

        loss = self.ce_loss(input, target.long())
        loss[mask] = 0.0
        loss = loss.sum() / (loss.numel() - num_mask_el)

        # revert dummy label
        target[mask] = 255

        return loss
