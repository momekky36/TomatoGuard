"""
dataset.py
----------
Builds train/val/test DataLoaders for the tomato leaf-disease ImageFolder
dataset, with the standard ImageNet normalization (needed for transfer
learning) and light augmentation on the training split.
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def get_transforms(image_size: int = IMAGE_SIZE):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2, image_size: int = IMAGE_SIZE):
    train_tf, eval_tf = get_transforms(image_size)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)

    # The real-world dataset is imbalanced (e.g. Yellow Leaf Curl Virus has
    # ~3x more images than Mosaic Virus). A weighted sampler keeps every
    # epoch class-balanced instead of letting majority classes dominate.
    class_counts = torch.bincount(torch.tensor(train_ds.targets))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[torch.tensor(train_ds.targets)]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes
