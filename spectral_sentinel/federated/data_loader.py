"""
Data Loading and Partitioning for Federated Learning

Non-IID data partitioning using Dirichlet distribution.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict
import os


def load_federated_data(dataset_name: str,
                       num_clients: int,
                       non_iid_alpha: float = 0.5,
                       data_dir: str = "./data") -> Tuple[List[Dataset], Dataset]:
    """
    Load and partition federated dataset.
    
    Args:
        dataset_name: 'mnist', 'cifar10', 'cifar100', 'femnist', 'tiny_imagenet'
        num_clients: Number of clients
        non_iid_alpha: Dirichlet concentration parameter (lower = more skew)
        data_dir: Data directory
        
    Returns:
        (train_datasets, test_dataset)
        - train_datasets: List of train datasets, one per client
        - test_dataset: Global test dataset
    """
    # Load raw dataset
    if dataset_name.lower() == 'mnist':
        train_dataset, test_dataset = load_mnist(data_dir)
    elif dataset_name.lower() == 'cifar10':
        train_dataset, test_dataset = load_cifar10(data_dir)
    elif dataset_name.lower() == 'cifar100':
        train_dataset, test_dataset = load_cifar100(data_dir)
    elif dataset_name.lower() == 'femnist':
        train_dataset, test_dataset = load_femnist(data_dir, num_clients)
        # FEMNIST is naturally partitioned, return as-is
        return train_dataset, test_dataset
    elif dataset_name.lower() == 'tiny_imagenet':
        train_dataset, test_dataset = load_tiny_imagenet(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Partition training data across clients (Non-IID)
    client_datasets = partition_data_non_iid(
        train_dataset,
        num_clients,
        non_iid_alpha
    )
    
    return client_datasets, test_dataset


def load_mnist(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset


def load_cifar10(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform_test
    )
    
    return train_dataset, test_dataset


def load_cifar100(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Load CIFAR-100 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = datasets.CIFAR100(
        data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        data_dir, train=False, download=True, transform=transform_test
    )
    
    return train_dataset, test_dataset


def partition_data_non_iid(dataset: Dataset,
                           num_clients: int,
                           alpha: float = 0.5) -> List[Subset]:
    """
    Partition dataset using Dirichlet distribution for Non-IID splits.
    
    For each class, sample from Dirichlet(α) to determine how to
    distribute that class's samples across clients.
    
    Lower α = more skewed (more Non-IID)
    Higher α = more uniform (closer to IID)
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        
    Returns:
        List of datasets, one per client
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Extract labels manually
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = {
        c: np.where(labels == c)[0]
        for c in range(num_classes)
    }
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute using Dirichlet
    for c in range(num_classes):
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        
        # Get indices for this class
        class_idx = class_indices[c]
        np.random.shuffle(class_idx)
        
        # Split according to proportions
        splits = (proportions * len(class_idx)).astype(int)
        splits[-1] = len(class_idx) - splits[:-1].sum()  # Adjust last to get exact count
        
        # Distribute to clients
        start = 0
        for client_id in range(num_clients):
            end = start + splits[client_id]
            client_indices[client_id].extend(class_idx[start:end])
            start = end
    
    # Create subset datasets
    client_datasets = [
        Subset(dataset, indices)
        for indices in client_indices
    ]
    
    # Print statistics
    print(f"\n📊 Data Partitioning (Non-IID, α={alpha}):")
    for i, client_dataset in enumerate(client_datasets):
        client_labels = labels[client_dataset.indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        print(f"  Client {i}: {len(client_dataset)} samples, "
              f"classes {unique.tolist()}, "
              f"dominant: {unique[np.argmax(counts)]}")
    
    return client_datasets


def compute_label_distribution(dataset: Dataset) -> Dict[int, int]:
    """
    Compute label distribution in dataset.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        Dict mapping labels to counts
    """
    if isinstance(dataset, Subset):
        # Get labels from subset
        if hasattr(dataset.dataset, 'targets'):
            all_labels = np.array(dataset.dataset.targets)
        else:
            all_labels = np.array(dataset.dataset.labels)
        labels = all_labels[dataset.indices]
    else:
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        else:
            labels = np.array(dataset.labels)
    
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def create_data_loader(dataset: Dataset,
                      batch_size: int = 32,
                      shuffle: bool = True) -> DataLoader:
    """
    Create data loader.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )


def load_femnist(data_dir: str = "./data", num_clients: int = 50) -> Tuple[List[Dataset], Dataset]:
    """
    Load FEMNIST (Federated EMNIST) dataset.
    
    FEMNIST has 62 classes (digits 0-9, lowercase a-z, uppercase A-Z).
    We'll use EMNIST ByClass as approximation.
    
    Args:
        data_dir: Data directory
        num_clients: Number of clients (default 50 for scaled-down version)
        
    Returns:
        (client_datasets, test_dataset)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,))  # EMNIST stats
    ])
    
    # Use EMNIST ByClass (62 classes)
    train_dataset = datasets.EMNIST(
        data_dir, split='byclass', train=True, download=True, transform=transform
    )
    test_dataset = datasets.EMNIST(
        data_dir, split='byclass', train=False, download=True, transform=transform
    )
    
    # Partition naturally by writer (simulate FEMNIST's natural partitioning)
    # For scaled-down version, we partition across num_clients using Dirichlet
    client_datasets = partition_data_non_iid(
        train_dataset,
        num_clients,
        alpha=0.3  # High heterogeneity for FEMNIST
    )
    
    print(f"\n📊 FEMNIST: {num_clients} clients, 62 classes")
    
    return client_datasets, test_dataset


def load_tiny_imagenet(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """
    Load Tiny ImageNet dataset (200 classes, 64x64 images).
    
    Note: This requires downloading Tiny ImageNet separately.
    For now, we'll use CIFAR-100 as a placeholder with resizing.
    
    Args:
        data_dir: Data directory
        
    Returns:
        (train_dataset, test_dataset)
    """
    # Transform to 64x64 for ViT-Small
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                           (0.2770, 0.2691, 0.2821))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                           (0.2770, 0.2691, 0.2821))
    ])
    
    # Use CIFAR-100 as placeholder (100 classes)
    train_dataset = datasets.CIFAR100(
        data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        data_dir, train=False, download=True, transform=transform_test
    )
    
    print(f"\n📊 Tiny ImageNet (CIFAR-100 placeholder): 100 classes, 64x64 images")
    
    return train_dataset, test_dataset


def compute_tv_distance(dataset: Dataset, reference_labels: np.ndarray) -> float:
    """
    Compute Total Variation (TV) distance between dataset and reference distribution.
    
    TV distance measures heterogeneity (0 = identical, 1 = completely different).
    
    Args:
        dataset: Client's dataset
        reference_labels: Reference label distribution (e.g., uniform or global)
        
    Returns:
        TV distance (0 to 1)
    """
    # Get client's label distribution
    if isinstance(dataset, Subset):
        if hasattr(dataset.dataset, 'targets'):
            all_labels = np.array(dataset.dataset.targets)
        else:
            all_labels = np.array(dataset.dataset.labels)
        client_labels = all_labels[dataset.indices]
    else:
        if hasattr(dataset, 'targets'):
            client_labels = np.array(dataset.targets)
        else:
            client_labels = np.array(dataset.labels)
    
    # Count occurrences
    client_counts = np.bincount(client_labels, minlength=len(np.unique(reference_labels)))
    reference_counts = np.bincount(reference_labels, minlength=len(np.unique(reference_labels)))
    
    # Normalize to probabilities
    client_prob = client_counts / client_counts.sum()
    reference_prob = reference_counts / reference_counts.sum()
    
    # TV distance = 0.5 * sum(|p - q|)
    tv = 0.5 * np.abs(client_prob - reference_prob).sum()
    
    return tv
