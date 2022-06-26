import argparse
import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from transforms import *
from torch.utils.data import DataLoader
import models


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="data")
    parser.add_argument('--weights', default="weights/ViT-S16.pth")
    parser.add_argument('--arch', default="vit_small")
    parser.add_argument('--patch_size', default=16)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--n_blocks', default=4)
    parser.add_argument('--batch_size', default=2)
    

    return parser.parse_args()

def train(loader, backbone, classifier, criterion, optimizer, n_blocks):
    backbone.eval()
    loss_l = []

    progress_bar = tqdm(total=len(loader))
    for img, segmentation in loader:
        optimizer.zero_grad()
        img = img.cuda()
        segmentation = segmentation.cuda()
        with torch.no_grad():
            intermediate_output = backbone.get_intermediate_layers(img, n_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1).detach()
        linear_output= classifier(output)
        #linear_output = torch.flatten(linear_output, start_dim=1)
        #segmentation = torch.flatten(segmentation, start_dim=1)

        loss = criterion(linear_output, segmentation.long())
        loss.backward()
        optimizer.step()
        loss_l.append(loss.item())
        progress_bar.update()
    return np.mean(np.array(loss_l))

class LinearClassifier(nn.Module):
    def __init__(self, dim, num_classes=22, img_size=224) :
        super(LinearClassifier, self).__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.linear = nn.Linear(dim, num_classes*img_size*img_size)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = self.linear(x)
        x = x.view(x.size(0), self.num_classes, self.img_size, self.img_size).contiguous()
        x = torch.softmax(x, dim=1)

        return x

def main(args):

    # Loading the backbone
    backbone = models.__dict__[args.arch](
        patch_size=args.patch_size,
        return_all_tokens=True,
    )

    state_dict = torch.load(args.weights)['state_dict']
    backbone.load_state_dict(state_dict)
    backbone = backbone.cuda()
    
        
    for param in backbone.parameters():
        param.requires_grad = False

    train_transform = Compose([
        RandomCrop(224),
        PILToTensor(),
        MergeContours(),
        RandomHorizontalFlip(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    n_blocks = args.n_blocks
    embed_dim = backbone.embed_dim * n_blocks
    classifier = LinearClassifier(embed_dim, img_size=224).cuda()
    optimizer = torch.optim.AdamW(classifier.parameters())

    for param in backbone.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    dataset = datasets.VOCSegmentation(root=args.root, image_set='train', download=False, transforms=train_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        mean_loss = train(loader, backbone, classifier, criterion, optimizer, n_blocks)
        print(f"For epoch number {epoch} --> Average Loss {mean_loss:.2f}")

def show_img(img, segmentation):
    img_pil = transforms.functional.to_pil_image(img)
    img_pil.show()
    seg_pil = transforms.functional.to_pil_image(segmentation)
    seg_pil.show()

if __name__ == '__main__':
    args = parser_args()
    main(args)
