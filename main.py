import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

# Hypothetical data: [mass, radius, distance from sun], target: [position]
data = np.array([
    [0.33, 4879, 57.9, 1],  # Mercury
    [4.87, 12104, 108.2, 2], # Venus
    [5.97, 12756, 149.6, 3], # Earth
    [0.642, 6792, 227.9, 4]  # Mars
])

X = data[:, :-1]
Y = data[:, -1]

n_qubits = X.shape[1]
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(weights, features):
    qml.templates.AngleEmbedding(features, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

weight_shapes = {"weights": (3, n_qubits)}
weights = np.random.random(size=(3, n_qubits))

def cost(weights, X, Y):
    predictions = np.array([quantum_circuit(weights, x) for x in X])
    loss = np.mean((predictions - Y)**2)
    return loss

opt = NesterovMomentumOptimizer(stepsize=0.1)

max_iter = 100
for it in range(max_iter):
    weights = opt.step(lambda v: cost(v, X, Y), weights)
    if it % 10 == 0:
        print(f"Iteration {it}, Cost: {cost(weights, X, Y)}")

predictions = np.array([quantum_circuit(weights, x) for x in X])
print("Predictions:", predictions)
print("Actual:", Y)
