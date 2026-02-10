"""Utility functions for data loading and preprocessing."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_cifar10_loaders(batch_size=32, val_split=0.1, data_dir='./data/raw'):
    """
    Get CIFAR-10 data loaders with train/val/test splits.
    
    Args:
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        data_dir: Directory to store/load the dataset
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Data augmentation and normalization for training
    # CIFAR-10 mean and std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Download and load training data
    train_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    # Split training data into train and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load test data
    test_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader


def get_class_names():
    """Get CIFAR-10 class names."""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']
