import json
import time
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from src.trainers.classical import train_cnn, train_svm, train_logreg
from src.trainers.quantum import train_vqc, train_qkernel

DATA_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
SEEDS = [42, 123, 999]

MODELS = {
    "cnn": train_cnn,
    "svm": train_svm,
    "logreg": train_logreg,
    "vqc": train_vqc,
    "qkernel": train_qkernel
}

results = {
    "dataset": "MNIST_0vs1",
    "models": {}
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_binary_mnist():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(
        root="datasets/mnist",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="datasets/mnist",
        train=False,
        download=True,
        transform=transform
    )

    # keep only digits 0 and 1
    train_indices = [i for i, (_, y) in enumerate(train_data) if y in [0, 1]]
    test_indices  = [i for i, (_, y) in enumerate(test_data)  if y in [0, 1]]

    train_subset = Subset(train_data, train_indices)
    test_subset  = Subset(test_data,  test_indices)

    return train_subset, test_subset

def subsample_dataset(dataset, fraction, seed):
    set_seed(seed)

    total_size = len(dataset)
    subset_size = int(total_size * fraction)

    indices = np.random.permutation(total_size)[:subset_size]
    return Subset(dataset, indices)

def run_experiments():
    train_full, test_set = load_binary_mnist()

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    for model_name, train_fn in MODELS.items():
        results["models"][model_name] = {}

        for frac in DATA_FRACTIONS:
            results["models"][model_name][str(frac)] = {
                "accuracy": [],
                "train_time": []
            }

            for seed in SEEDS:
                set_seed(seed)

                train_subset = subsample_dataset(train_full, frac, seed)
                train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

                start = time.time()
                acc = train_fn(train_loader, test_loader)
                end = time.time()

                results["models"][model_name][str(frac)]["accuracy"].append(acc)
                results["models"][model_name][str(frac)]["train_time"].append(end - start)

def save_results():
    with open("results/mnist/lowdata.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_experiments()
    save_results()
