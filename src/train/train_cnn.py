import torch
import torch.nn as nn
import torch.optim as optim
from src.data.mnist_loader import get_mnist_binary_loaders
from src.models.cnn import SimpleCNN
from src.eval.evaluate import evaluate

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_binary_loaders(
        batch_size=32,
        train_frac=1.0
    )

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}, Loss={total_loss:.3f}, Test Acc={acc:.3f}")

if __name__ == "__main__":
    train()