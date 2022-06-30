import argparse
from datetime import datetime
from random import shuffle
import torch
import torch.nn as nn
import wandb


from utils import mIoU
from dataloader import PartialDatasetVOC
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as Fun
from tqdm import tqdm
from transforms import *
from torch.utils.data import DataLoader
import models
from torch.utils.tensorboard import SummaryWriter


CLASS_LABELS = {
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
}

global_step = 0

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="data")
    parser.add_argument('--weights', default="weights/ViT-S16.pth")
    parser.add_argument('--arch', default="vit_small")
    parser.add_argument('--patch_size', default=16)
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--n_blocks', default=4)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument("--init_lr", default=1e-3)
    parser.add_argument("--min_lr", default=1e-5)
    parser.add_argument("--percentage", default=0.1)
    parser.add_argument("--log_folder", default="logs/")
    

    return parser.parse_args()

def train(loader, backbone, classifier, criterion, optimizer, n_blocks, scheduler):
    global global_step
    backbone.eval()
    loss_l = []

    progress_bar = tqdm(total=len(loader))
    for img, segmentation in loader:
        optimizer.zero_grad()
        img = img.cuda()
        segmentation = segmentation.cuda()
        with torch.no_grad():
            intermediate_output = backbone.get_intermediate_layers(img, n_blocks)
            output = torch.cat([x[:, 1:] for x in intermediate_output], dim=-1).detach()
        linear_output= classifier(output)

        loss = criterion(linear_output, segmentation.long())
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_l.append(loss.item())
        progress_bar.update()
        global_step += 1

    return np.mean(np.array(loss_l))


def validate(loader, backbone, classifier, criterion, n_blocks):
    backbone.eval()
    classifier.eval()
    val_loss = []
    miou_arr = []
    random_pic_select = np.random.randint(len(loader))
    with torch.no_grad():
        for idx , (img, segmentation) in enumerate(loader):
            img = img.cuda()
            segmentation = segmentation.cuda()
            intermediate_output = backbone.get_intermediate_layers(img, n_blocks)
            output = torch.cat([x[:, 1:] for x in intermediate_output], dim=-1).detach()
            linear_output = classifier(output)
            loss = criterion(linear_output, segmentation.long())
            val_loss.append(loss.item())
            miou = mIoU(linear_output,segmentation)
            miou_arr.append(miou.item())
            if random_pic_select==idx:
                print("Adding Image Example to Logger")
                pred_segmentation = wandb.Image(img[0],
                    masks={
                        "predictions": {
                            "mask_data": torch.argmax(linear_output, dim=1).squeeze(0).cpu().numpy(),
                            "class_labels": CLASS_LABELS
                            }
                        })
                gt_segmentation = wandb.Image(img[0],
                    masks={
                        "ground_truth": {
                            "mask_data": segmentation.squeeze(0).cpu().numpy(),
                            "class_labels": CLASS_LABELS
                            }
                        })
                wandb.log({
                    "Pred Segmentation": pred_segmentation,
                    "GT Segmentation": gt_segmentation},
                    step=global_step)

    return np.mean(np.array(miou_arr)), np.mean(np.array(val_loss))

class ConvLinearClassifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=embed_dim, out_channels=n_classes, kernel_size=1)

    def forward(self,x):
        bs, h_sqrt , ch= x.shape
        h = int(np.sqrt(h_sqrt))
        x =  x.view(bs,ch, h, h)
        x = self.conv1(x)
        x = Fun.interpolate(x, size=[224, 224], mode="bilinear", align_corners=True)
        return x


def main(args):
    config = {
        "arch": args.arch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.init_lr,
        "patch_size": args.patch_size,
        "number_blocks": args.n_blocks,
        "percentage_train_labels": args.percentage,
    }
    
    wandb.init(
        project="iBot",
        entity="dl_lab_enjoyers",
        name=datetime.now().strftime('%m.%d.%Y-%H:%M:%S'),
        config=config)

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
    classifier = ConvLinearClassifier(embed_dim, n_classes=21).cuda()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.init_lr)

    
    criterion = nn.CrossEntropyLoss()
    ## TRAINING DATASET ##
    train_dataset = PartialDatasetVOC(percentage = args.percentage, root=args.root, image_set='train', download=False, transforms=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    ## TRAINING DATASET ##
    val_dataset = datasets.VOCSegmentation(root=args.root, image_set='val', download=False, transforms=train_transform)
    val_loader = DataLoader(val_dataset, batch_size=1)
    ####################################################
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    ####################################################

    for param in backbone.parameters():
        param.requires_grad = False

    ############## TRAINING LOOP #######################

    for epoch in range(args.epochs):
        mean_loss = train(train_loader, backbone, classifier, criterion, optimizer, n_blocks, lr_scheduler)
        print(f"For epoch number {epoch} --> Average Loss {mean_loss:.2f}")
        wandb.log({"training_loss": mean_loss}, step=global_step)
        if epoch % 10 == 0:
            miou, loss = validate(val_loader, backbone, classifier, criterion, n_blocks)
            print(f"Validation for epoch {epoch}: Average mIoU {miou}, Average Loss {loss}")
            wandb.log({
                "val_loss": loss,
                "val_miou": miou
            })
        if epoch == 55:
            x = 3

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    iters = np.arange(epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return schedule

def show_img(img):
    img_pil = transforms.functional.to_pil_image(img)
    img_pil.save("dum1.png")


if __name__ == '__main__':
    args = parser_args()
    main(args)
