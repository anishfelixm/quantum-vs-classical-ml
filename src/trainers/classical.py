import torch
import torch.nn as nn
import torch.optim as optim
from src.data.mnist_loader import get_mnist_binary_loaders
from src.models.cnn import SimpleCNN
from src.eval.evaluate import evaluate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


def train_logreg(train_loader, test_loader):
    X_train, y_train = [], []
    X_test, y_test = [], []

    for x, y in train_loader:
        x = x.view(x.size(0), -1)
        X_train.append(x.cpu().numpy())
        y_train.append(y.cpu().numpy())

    for x, y in test_loader:
        x = x.view(x.size(0), -1)
        X_test.append(x.cpu().numpy())
        y_test.append(y.cpu().numpy())
    
    X_train = np.concatenate(X_train, axis = 0)
    y_train = np.concatenate(y_train, axis = 0)
    X_test  = np.concatenate(X_test, axis = 0)
    y_test  = np.concatenate(y_test, axis = 0)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

def train_svm(*args):
    return 0.0

def train_cnn(*args):
    return 0.0

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
