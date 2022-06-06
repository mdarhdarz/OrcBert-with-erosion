import os
import warnings
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

warnings.filterwarnings('ignore')


class Erode(object):
    def __init__(self, kernel_size=9, stride=1, padding=4, p=0.5):
        self.p = p
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def __call__(self, img):
        if np.random.rand() < self.p:
            img_ = -self.max_pool(-img)
            return img_
        else:
            return img


def get_loader(args, data_transforms=None):
    if data_transforms is None:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(60, fill=255, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                # Erode(),
                # transforms.Normalize([0.84, 0.84, 0.84], [0.32, 0.32, 0.32])
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.84, 0.84, 0.84], [0.32, 0.32, 0.32])
            ]),
        }

    image_datasets = {datatype: datasets.ImageFolder(
        os.path.join(args.data_dir, datatype),
        data_transforms[datatype])
        for datatype in ['train', 'test']}
    dataloaders = {datatype: DataLoader(image_datasets[datatype],
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.workers)
                   for datatype in ['train', 'test']}

    return dataloaders
