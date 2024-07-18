import streamlit as st
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

# Streamlit UI
st.title("Quantum Machine Learning for Solar System Analysis")
st.write("Predicting planetary positions using Quantum Machine Learning")

# Display the data
st.write("### Input Data")
st.write("Features: [mass, radius, distance from sun]")
st.write(data)

# Display predictions
st.write("### Predictions vs Actual")
st.write(f"Predictions: {predictions}")
st.write(f"Actual: {Y}")

# User input for new data
st.write("### Predict New Data")
mass = st.number_input("Mass", min_value=0.0, value=5.97)
radius = st.number_input("Radius", min_value=0.0, value=12756.0)
distance = st.number_input("Distance from Sun", min_value=0.0, value=149.6)

new_data = np.array([mass, radius, distance])

if st.button("Predict"):
    new_prediction = quantum_circuit(weights, new_data)
    st.write(f"Predicted Position: {new_prediction}")

# Button to retrain the model
if st.button("Retrain Model"):
    weights = np.random.random(size=(3, n_qubits))
    for it in range(max_iter):
        weights = opt.step(lambda v: cost(v, X, Y), weights)
        if it % 10 == 0:
            st.write(f"Iteration {it}, Cost: {cost(weights, X, Y)}")
    predictions = np.array([quantum_circuit(weights, x) for x in X])
    st.write("Model retrained!")
    st.write(f"New Predictions: {predictions}")
