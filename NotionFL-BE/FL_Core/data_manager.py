import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np


# Defining transforms for MNIST
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Defining transforms for CIFAR-10
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_data_loaders(dataset_name, batch_size=64, train_shuffle=True, test_shuffle=True):
    """
    Returns data loaders for the given dataset
    
    Args:
        dataset_name (str) : Supported datasets are 'MNIST' and 'CIFAR10'
        batch_size (int) : Number of data batches to train
        train_shuffle (bool) : Whether to shuffle the training data
        test_shuffle (bool) : Whether to shuffle the test data     
    
    Returns:
        tuple: training_loader, test_loader
    
    """
    if dataset_name == 'MNIST':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
    elif dataset_name == 'CIFAR10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=train_shuffle)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=test_shuffle)

    return train_loader, test_loader

def split_client_data(dataset, num_clients, batch_size=64):
    """
        Splits dataset for multiple clients:
        ensures data distribution remains uneven (reflecting real-world scenarios), 
        all data points are still utilized and fairly distributed among clients.
    """
    total_data_points = len(dataset)
    indices = torch.randperm(total_data_points).tolist()
    # Calculate split sizes for each client
    split_sizes = [total_data_points // num_clients + (1 if i < total_data_points % num_clients else 0) for i in range(num_clients)]
    client_data_loaders = {}
    index = 0
    
    for i in range(num_clients):
        client_size = split_sizes[i]
        client_indices = indices[index:index+client_size]
        client_data_loaders[f'client_{i}'] = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(client_indices))
        index += client_size

    return client_data_loaders
