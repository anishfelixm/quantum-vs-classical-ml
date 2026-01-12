import torch
import pennylane as qml
import torch.nn as nn
import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # Angle encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Variational layers
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return qml.expval(qml.PauliZ(0))


class QuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            0.01 * torch.randn(1, n_qubits, 3, dtype=torch.float64)
            )
        
    def forward(self, x):
        return quantum_circuit(x, self.weights)