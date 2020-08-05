from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import math
import random
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

def random_affine(img, degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4


    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    return img
class Data(nn.Module):
    def __init__(self, phase, transform, path=''):
        super().__init__()
        self.transform = transform
        self.images_path = []
        self.phase = phase
        self.img_size = 224
        self.mosaic = True

        for file in os.listdir(os.path.join(path, "0")):
            if ".DS_Store" in file:
                continue
            self.images_path.append(os.path.join(path, "0", file))
        for file in os.listdir(os.path.join(path, "1")):
            if ".DS_Store" in file:
                continue
            self.images_path.append(os.path.join(path, "1", file))

    def __getitem__(self, id):
        if self.mosaic:
            image, label = self.load_mosaic(id)
        else:
            image = Image.open(self.images_path[id])
            image = self.transform(image)
            cls = int(self.images_path[id].split('/')[-2])
            label = [0] * 2
            label[cls] = 1

        return image, torch.Tensor(label).float()

    def __len__(self):
        return len(self.images_path)


    def load_mosaic(self, index):
        # loads images in a mosaic

        s = self.img_size // 2
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y

        cls = int(self.images_path[index].split('/')[-2])
        indices = []
        for index_aother in range(index, len(self.images_path)):
            if int(self.images_path[index_aother].split('/')[-2]) == cls:
                indices.append(index_aother)
            if len(indices) == 4:
                break
        if len(indices) < 4:
            for index_aother in range(index):
                if int(self.images_path[index_aother].split('/')[-2]) == cls:
                    indices.append(index_aother)
                if len(indices) == 4:
                    break

        for i, index in enumerate(indices):
            # Load image
            img, h, w = self.load_image(index)
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        img4 = random_affine(img4)  # border to remove
        label4 = [0] * 2
        label4[cls] = 1

        return img4, label4

    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        path = self.images_path[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        print("r:", r)
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, img.shape[0], img.shape[1]  # img, hw_original, hw_resized




if __name__ == "__main__":
    import os
    import cv2
    images_path = "data_images/train"

    trans_train = transforms.Compose([transforms.ToTensor()])
    train_set = Data("train", trans_train, images_path)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=2, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True,
    )
    for i, (images, labels) in enumerate(train_loader):
        images = images.to("cpu")
        labels = labels.to("cpu")
        labels = labels.squeeze()
        _, labels = torch.max(labels, 1)
        print("labels:", labels)
        print("images:", images.shape)

        case = np.array(images[0, :])
        cv2.namedWindow('case', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('case', case)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()

        break
