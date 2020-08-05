from PIL import Image
import os
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import torchvision
import torchvision.transforms as transforms
import random
import PIL.Image as Image

def resnet18():
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)
    return model

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, input, label):
        loss = self.criterion(input, label)
        return torch.mean(loss)

class Data(nn.Module):
    def __init__(self, phase, transform, path=''):
        super().__init__()
        self.transform = transform
        self.images_path = []
        self.phase = phase
        for file in os.listdir(os.path.join(path, "0")):
            if ".DS_Store" in file:
                continue
            self.images_path.append(os.path.join(path, "0", file))
        for file in os.listdir(os.path.join(path, "1")):
            if ".DS_Store" in file:
                continue
            self.images_path.append(os.path.join(path, "1", file))
    def __getitem__(self, id):
        image = Image.open(self.images_path[id])
        image = self.transform(image)
        cls = int(self.images_path[id].split('/')[-2])
        label = [0] * 2
        label[cls] = 1
        return image, torch.Tensor(label).float()
    def __len__(self):
        return len(self.images_path)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def run_epoch(
        phase, model, criterion, optimizer, data_loader,
        epoch, args):
    '''
    Run an epoch
    '''
    if phase == 'train':
        model.train()
    else:
        model.eval()
    # Create recorder
    if not os.path.exists(os.path.join(args.out_dir, 'net', phase)):
        os.mkdir(os.path.join(args.out_dir, 'net', phase))
    recorder = open(
        os.path.join(args.out_dir, 'net', phase, str(epoch) + '.txt'), 'w'
    )
    # Criterions
    avg_loss = 0
    c = 0
    n = 0
    # Start training/valuating
    for i, (images, labels_onehoe) in enumerate(data_loader):
        images = images.to(args.device)
        labels_onehoe = labels_onehoe.to(args.device)
        labels = labels_onehoe.squeeze()
        _, labels = torch.max(labels, 1)
        if phase == 'train':
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cpu()
                target_a = labels_onehoe
                target_b = labels_onehoe[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                output = model(images)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                output = model(images)
                loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(images)
            loss = criterion(output, labels)
        _, preds = torch.max(output, 1)
        preds = preds.cpu().data.numpy().squeeze()
        labels = labels.cpu().data.numpy().squeeze()
        loss = loss.item()
        # print("loss.item()", loss)
        avg_loss += loss

    # Calculate criterions
    avg_loss /= len(data_loader)
    return avg_loss

def train(args):
    # Dataloader
    trans_train = [
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((-45, 45)),
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    trans_train = transforms.Compose(trans_train)
    trans_val = [
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    trans_val = transforms.Compose(trans_val)

    train_set = Data('train', trans_train, path=args.train)
    val_set = Data('val', trans_val, path=args.val)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=16,
        pin_memory=True, drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=16,
        pin_memory=True, drop_last=False,
    )

    # Model, loss function and optimizer
    model = resnet18()
    model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    loss = Loss().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    best_loss = 9999
    for epoch in range(args.epochs):
        train_loss = run_epoch(
            'train', model, loss, optimizer, train_loader,
            epoch, args
        )
        val_loss = run_epoch(
            'val', model, loss, None, val_loader, epoch,
            args
        )
        if val_loss < best_loss:
            best_loss = val_loss
        scheduler.step(val_loss)
        print("train_loss:", train_loss)
        print("val_loss:", val_loss)

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is', args.device)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    train(args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Augumentation')
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--phase', type=str, help='train or test')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--crop-size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--cutmix_prob', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--out_dir', type=str, default='data_images/output')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--train', type=str, default='data_images/train')
    parser.add_argument('--val', type=str, default='data_images/train')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    args = parse_args()
    main(args=args)
