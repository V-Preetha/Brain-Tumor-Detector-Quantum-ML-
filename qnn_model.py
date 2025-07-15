import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

# Set number of qubits based on image size (3x3 = 9)
n_qubits = 16
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs):
    for i in range(n_qubits):
        qml.RY(inputs[i] * torch.pi, wires=i)

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    for i in range(n_qubits):
        qml.RZ(inputs[i] * torch.pi, wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]  # <-- 9 outputs!

# Define Torch layer to wrap the quantum circuit
class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
            out = quantum_circuit(x[i])
            outputs.append(torch.tensor(out))
        return torch.stack(outputs)

# Define the full hybrid model
class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnn = QuantumLayer()

        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.qnn(x)
        return self.classifier(x.float())

