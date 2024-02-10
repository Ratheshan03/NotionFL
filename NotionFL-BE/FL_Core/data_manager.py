# FL_Core/data_manager.py
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

def get_mnist_data_loaders(batch_size=64, train_shuffle=True, test_shuffle=True):
    # Load MNIST dataset for training and evaluation
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=train_shuffle)
    test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=test_shuffle)

    return train_loader, test_loader

def split_data_for_clients(mnist_trainset, num_clients, batch_size=64):
    # Get the total number of data points
    total_data_points = len(mnist_trainset)
    # Generate a random permutation of indices
    indices = torch.randperm(total_data_points).tolist()
    
    # Calculate the split size for each client, ensuring all data is used
    split_sizes = [total_data_points // num_clients + (1 if i < total_data_points % num_clients else 0) for i in range(num_clients)]
    
    # Randomly assign indices to each client
    client_indices = [indices[sum(split_sizes[:i]):sum(split_sizes[:i+1])] for i in range(num_clients)]
    
    # Create a data loader for each client using SubsetRandomSampler
    client_data_loaders = {f'client_{client_id}': DataLoader(mnist_trainset, batch_size=batch_size, sampler=SubsetRandomSampler(client_idx))
                           for client_id, client_idx in enumerate(client_indices)}
    
    return client_data_loaders

def create_client_data_loaders(client_datasets, batch_size=64):
    # Create data loaders for each client's dataset
    client_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in client_datasets]

    return client_loaders
