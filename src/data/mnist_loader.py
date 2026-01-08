import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_mnist_binary_loaders(batch_size=32, train_frac=1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="datasets/mnist",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="datasets/mnist",
        train=False,
        download=True,
        transform=transform
    )

    def filter_binary(dataset):
        indices = [i for i, (_, y) in enumerate(dataset) if y in [0, 1]]
        return Subset(dataset, indices)

    train_dataset = filter_binary(train_dataset)
    test_dataset = filter_binary(test_dataset)

    if train_frac < 1.0:
        n = int(len(train_dataset) * train_frac)
        indices = np.random.choice(len(train_dataset), n, replace=False)
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
