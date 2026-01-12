import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.models.qml_vqc import QuantumClassifier

def preprocess(x):
    # x: [1, 28, 28]
    x = F.avg_pool2d(x, kernel_size=7)  # → [1, 4, 4]
    x = x.view(-1)[:4]                  # → 4 features
    return x.to(torch.float64)

def get_binary_mnist(batch_size=1, limit=200):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root="datasets/mnist",
        train=True,
        download=True,
        transform=transform
    )

    indices = [i for i, (_, y) in enumerate(dataset) if y in [0, 1]]
    indices = indices[:limit]  # keep dataset small for QML

    dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train():
    device = torch.device("cpu")  # QML runs on CPU
    model = QuantumClassifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    loader = get_binary_mnist()

    epochs = 10
    for epoch in range(epochs):
        correct = 0
        total = 0
        loss_sum = 0

        for x, y in loader:
            x = preprocess(x).to(device)
            y = y.to(torch.float64).to(device)
            optimizer.zero_grad()


            output = model(x).unsqueeze(0)   # shape [1]
            target = (2 * y - 1) # shape [1]

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            pred = 1 if output.item() > 0 else 0
            correct += int(pred == y.item())
            total += 1

        acc = correct / total
        print(f"Epoch {epoch+1}, Loss={loss_sum:.3f}, Acc={acc:.3f}")

if __name__ == "__main__":
    train()
