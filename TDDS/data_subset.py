import numpy as np
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from multiprocessing import Pool
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

########################################################################################################################
#  Load Data
########################################################################################################################

def load_cifar10_sub(args, data_mask, sorted_score):
    """
    Load CIFAR10 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
    train_data.targets = z

    subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
    data_set = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader


import random


def load_cifar100_sub(args, data_mask=None, sorted_score=None):
    """
    Load CIFAR100 dataset with specified transformations and allow random sample removal.
    """
    print('Loading CIFAR100... ', end='')
    time_start = time.time()

    # Normalization parameters
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load CIFAR-100 training dataset
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)

    # Option 1: Randomly drop samples
    if args.pruning_methods == -1:
        total_samples = len(train_data.targets)
        num_keep = int((1 - args.subset_rate) * total_samples)  # Keep a percentage of samples
        random_indices = random.sample(range(total_samples), num_keep)  # Randomly select indices to keep

        # Construct the new targets list with label and a fixed score of 1
        z = [[train_data.targets[i], 1] for i in range(total_samples)]  # Assign a fixed score of 1
        train_data.targets = z

        # Create a subset using random indices
        train_data = torch.utils.data.Subset(train_data, random_indices)
    elif args.pruning_methods == 2:
        # Option 2: Use mask and score for filtering
        if data_mask is not None and sorted_score is not None:
            score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))  # Standardize scores
            z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
            train_data.targets = z
            subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
            train_data = torch.utils.data.Subset(train_data, subset_mask)

    elif args.pruning_methods == 3:
        # Option 2: Use mask and score for filtering
        if data_mask is not None and sorted_score is not None:
            # score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))  # Standardize scores
            score = np.ones_like(sorted_score, dtype=np.float32)
            z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
            train_data.targets = z
            subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
            train_data = torch.utils.data.Subset(train_data, subset_mask)

    elif args.pruning_methods == 4:
        # Option 2: Use mask and score for filtering
        if data_mask is not None and sorted_score is not None:
            # score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))  # Standardize scores
            score = np.ones_like(sorted_score, dtype=np.float32)
            z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
            train_data.targets = z
            subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
            train_data = torch.utils.data.Subset(train_data, subset_mask)

    elif args.pruning_methods == 5:
        # Option 2: Use mask and score for filtering
        if data_mask is not None and sorted_score is not None:
            score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))  # Standardize scores
            # score = np.ones_like(sorted_score, dtype=np.float32)
            z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
            train_data.targets = z
            subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
            train_data = torch.utils.data.Subset(train_data, subset_mask)

    # Training DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    # Data transformation for test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load CIFAR-100 test dataset
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    return train_loader, test_loader

def load_cifar100_sub_old(args, data_mask, sorted_score):
    """
    Load CIFAR100 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
    train_data.targets = z

    subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
    data_set = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader
