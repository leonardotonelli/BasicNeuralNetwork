import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data(batch_size=32):
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Download and load the training data
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    # Create DataLoader for train and test datasets
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader