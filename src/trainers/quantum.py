import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.models.qml_vqc import QuantumClassifier


def train_vqc(*args):
    return 0.0

def train_qkernel(*args):
    return 0.0
