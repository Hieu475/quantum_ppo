import torch
import pennylane as qml

n_qubits = 2
try:
    dev = qml.device("lightning.gpu", wires=n_qubits)
except qml.DeviceError:
    dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit(inputs, weights):
    # inputs has shape (n_qubits, batch)
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

inputs = torch.randn(3, 2)
weights = torch.randn(2, requires_grad=True)

print("Original:")
out = []
for i in range(inputs.shape[0]):
    out.append(torch.stack(list(circuit(inputs[i], weights))))
print(torch.stack(out))

print("Broadcasting with inputs.T:")
inputs_T = inputs.T
res = circuit(inputs_T, weights)
print(torch.stack(list(res), dim=1))

print("Backward test:")
loss = torch.stack(list(circuit(inputs_T, weights)), dim=1).sum()
loss.backward()
print("Grads:", weights.grad)
