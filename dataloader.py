from torchvision import datasets


class DatasetVOC(datasets.VOCSegmentation):
    def __init__(self, percentage, root, year='2012', image_set='train', download=False, transform=None, target_transform=None, transforms=None):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        self.percentage = percentage
        
        self.images = self.images[:int(percentage*len(self.images))]
        self.masks = self.masks[:int(percentage*len(self.masks))]
        