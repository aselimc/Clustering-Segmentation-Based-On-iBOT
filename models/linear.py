import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from tqdm import tqdm

from . import _BaseSegmentator
from utils.scheduler import WarmStartCosineAnnealingLR


class LinearSegmentator(_BaseSegmentator):

    def __init__(self, backbone,
                 logger,
                 epochs=100,
                 warmup_epochs=10,
                 lr=1e-3,
                 eval_freq=10,
                 **kwargs):
        
        super(LinearSegmentator, self).__init__(backbone, logger, **kwargs)

        self.fc = nn.Linear(self.embed_dim, self.num_classes).to(device=self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.fc.parameters(),
                                   lr=lr,
                                   momentum=0.9,
                                   weight_decay=0)
        self.lr_scheduler = WarmStartCosineAnnealingLR(self.optimizer, epochs, warmup_epochs, min_lr=0)
        self.eval_freq = eval_freq
    
    def forward(self, image):
        bs = image.size(0)
        feat = self._extract_feature(image)

        logits = self.fc(feat)
        pred = torch.argmax(logits, dim=1)
        pred = pred.view(bs, self.num_patches)
        pred = pred.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        pred = pred.expand(bs, self.num_patches, 1, self.patch_size, self.patch_size)

        nrows = self.img_size // self.patch_size
        pred = [make_grid(p, nrows, padding=0) for p in pred]
        pred = torch.stack(pred)[:, 0]
        pred = self.smooth(pred)

        return pred

    def fit(self, loader):
        train_features, train_labels = self._transform_data(loader)
        train_features, train_labels = self._balance_class(train_features, train_labels)

        # create batches
        num_samples = train_features.size(0)
        bs = loader.batch_size * self.num_patches
        num_iter = math.ceil(num_samples / bs)
        train_features = torch.chunk(train_features, dim=0, chunks=num_iter)
        train_labels = torch.chunk(train_labels, dim=0, chunks=num_iter)

        loss_meter = []

        print("\nFit Linear...")
        for n in range(self.epochs):
            print(f"\nEpoch: {n}/{self.epochs}")
            progress_bar = tqdm(total=num_iter)
            for it in range(num_iter):
                image = train_features[it].to(device=self.device)
                target = train_labels[it].to(device=self.device)
                target = target.long()
                self.optimizer.zero_grad()

                logits = self.fc(image)
                logits = logits.unsqueeze(2).expand(-1, self.num_classes, self.patch_size**2)
                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                loss_meter.append(loss.item())
                if (it % self.eval_freq) == 0 or (it == num_iter-1):
                    self.logger.log_scalar({
                        "ce_loss": np.mean(loss_meter)
                    }, step=it+n*num_iter)

                progress_bar.update()

    @property
    def epochs(self):
        return self.lr_scheduler.epochs
