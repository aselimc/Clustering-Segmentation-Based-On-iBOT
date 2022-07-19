from torchvision import datasets


class PartialDatasetVOC(datasets.VOCSegmentation):
    def __init__(self, percentage, root, year='2012', image_set='train', download=False, transform=None, target_transform=None, transforms=None):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        self.percentage = percentage
        if percentage < 1:
            self.images = self.images[:int(percentage*len(self.images))]
            self.targets = self.targets[:int(percentage*len(self.targets))]
