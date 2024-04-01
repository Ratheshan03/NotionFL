# FL_Core/data_manager.py
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np

# Define transforms for MNIST
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Define transforms for CIFAR-10
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 mean and std
])

def get_data_loaders(dataset_name, batch_size=64, train_shuffle=True, test_shuffle=True):
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

def split_data_for_clients(dataset, num_clients, batch_size=64):
    total_data_points = len(dataset)
    indices = torch.randperm(total_data_points).tolist()
    split_sizes = [total_data_points // num_clients + (1 if i < total_data_points % num_clients else 0) for i in range(num_clients)]
    client_indices = [indices[sum(split_sizes[:i]):sum(split_sizes[:i+1])] for i in range(num_clients)]

    client_data_loaders = {
        f'client_{client_id}': DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(client_idx))
        for client_id, client_idx in enumerate(client_indices)
    }

    return client_data_loaders

# Function to create data loaders can remain as it is
def create_client_data_loaders(client_datasets, batch_size=64):
    client_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]
    return client_loaders
