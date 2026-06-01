import torch
import pennylane as qml

n_qubits = 2
try:
    dev = qml.device("lightning.gpu", wires=n_qubits)
except qml.DeviceError:
    dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

inputs = torch.randn(3, 4, requires_grad=True)
weights = torch.randn(2, requires_grad=True)

try:
    res = circuit(inputs, weights)
    print("Amplitude Broadcasting Passed:")
    print(torch.stack(list(res), dim=1))
    
    loss = torch.stack(list(res)).sum()
    loss.backward()
    print("Backward Passed. Grads:", weights.grad)
except Exception as e:
    print("Error:", e)
