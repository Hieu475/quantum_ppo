import torch
import pennylane as qml

n_qubits = 2
try:
    dev = qml.device("lightning.gpu", wires=n_qubits)
except qml.DeviceError:
    dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit(inputs, weights):
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    qml.RX(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

inputs = torch.randn(3, 2, requires_grad=True)
weights = torch.randn(2, requires_grad=True)

try:
    def qnode_wrapper(inp, w):
        return torch.stack(list(circuit(inp, w)))
    
    res = torch.vmap(qnode_wrapper, in_dims=(0, None))(inputs, weights)
    print("Forward passed.")
    loss = res.sum()
    loss.backward()
    print("Backward passed. Grads:", weights.grad)
except Exception as e:
    print("Error:", e)
