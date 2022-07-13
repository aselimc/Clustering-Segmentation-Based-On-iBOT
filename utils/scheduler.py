import math
import warnings

import torch.optim.lr_scheduler as lr_scheduler


class WarmStartCosineAnnealingLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, epochs, warmup_epochs=0, min_lr=0.0, last_epoch=-1, verbose=False):
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.T_max = epochs - warmup_epochs
        self.eta_min = min_lr

        super(WarmStartCosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        epoch = self.last_epoch - self.warmup_epochs

        # linear warmup
        if epoch < 0:
            return [(self.last_epoch + 1) * base_lr / self.warmup_epochs
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]

        if epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos((epoch) * math.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        elif (epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * epoch / self.T_max)) /
                (1 + math.cos(math.pi * (epoch - 1) / self.T_max)) * (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]
