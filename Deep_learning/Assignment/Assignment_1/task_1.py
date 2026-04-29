# %% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
# %% LOAD DATA
df = pd.read_csv("heart-disease.csv")

# Define feature matrix (X) and target (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1).astype(float)  # Ensure numerical stability

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% HYPERPARAMETERS
INPUT_SIZE = X_train.shape[1]  # Number of features
HIDDEN_SIZE = 5                # Hidden layer neurons
OUTPUT_SIZE = 1                # Output layer (binary classification)
LEARNING_RATE = 0.01           # Learning rate
BATCH_SIZE = 128               # Batch size
EPOCHS = 500                    # Number of epochs

# %% INITIALIZE PARAMETERS
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)  # Reproducibility
    return {
        "W1": np.random.uniform(-1, 1, (hidden_size, input_size)),
        "b1": np.zeros((hidden_size, 1)),
        "W2": np.random.uniform(-1, 1, (output_size, hidden_size)),
        "b2": np.zeros((output_size, 1))
    }

params = initialize_parameters(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# %% ACTIVATION FUNCTIONS
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    return A * (1 - A)

# %% FORWARD PROPAGATION
def forward_propagation(X, params):
    Z1 = np.dot(params["W1"], X.T) + params["b1"]
    A1 = sigmoid(Z1)
    Z2 = np.dot(params["W2"], A1) + params["b2"]
    A2 = sigmoid(Z2)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

# %% LOSS FUNCTION
def compute_loss(A2, y):
    m = y.shape[0]
    epsilon = 1e-10  # Avoid log(0)
    return -np.mean(y.T * np.log(A2 + epsilon) + (1 - y.T) * np.log(1 - A2 + epsilon))

# %% BACKPROPAGATION
def backpropagation(X, y, params, cache):
    m = X.shape[0]
    
    # Compute gradients
    dZ2 = cache["A2"] - y.T
    dW2 = (1 / m) * np.dot(dZ2, cache["A1"].T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(params["W2"].T, dZ2) * sigmoid_derivative(cache["A1"])
    dW1 = (1 / m) * np.dot(dZ1, X)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

# %% TRAINING FUNCTION
def train(X_train, y_train, X_test, y_test, params, learning_rate, batch_size, epochs):
    m = X_train.shape[0]
    loss_train, acc_train, loss_test, acc_test = [], [], [], []

    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        X_train, y_train = X_train[permutation], y_train[permutation]

        for i in range(0, m, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward propagation
            cache = forward_propagation(X_batch, params)

            # Backpropagation
            gradients = backpropagation(X_batch, y_batch, params, cache)

            # Update parameters
            for key in params:
                params[key] -= learning_rate * gradients["d" + key]

        # Evaluate on training and test sets
        train_cache = forward_propagation(X_train, params)
        test_cache = forward_propagation(X_test, params)

        loss_train.append(compute_loss(train_cache["A2"], y_train))
        loss_test.append(compute_loss(test_cache["A2"], y_test))

        acc_train.append(np.mean((train_cache["A2"] > 0.5).astype(int).T == y_train))
        acc_test.append(np.mean((test_cache["A2"] > 0.5).astype(int).T == y_test))

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss_train[-1]:.4f}, Train Acc = {acc_train[-1]:.4f} | Test Loss = {loss_test[-1]:.4f}, Test Acc = {acc_test[-1]:.4f}")

    return params, loss_train, acc_train, loss_test, acc_test





# %% TRAIN MODEL
params, loss_train, acc_train, loss_test, acc_test = train(
    X_train, y_train, X_test, y_test, params, LEARNING_RATE, BATCH_SIZE, EPOCHS
)

# %% PLOTTING FUNCTION
def plot_metrics(loss_train, loss_test, acc_train, acc_test):
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(loss_train, label="Train Loss", color='blue')
    plt.plot(loss_test, label="Test Loss", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Model Loss Over Epochs")

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(acc_train, label="Train Accuracy", color='blue')
    plt.plot(acc_test, label="Test Accuracy", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Model Accuracy Over Epochs")

    plt.show()
    
    
    

# %% PLOT RESULTS
plot_metrics(loss_train, loss_test, acc_train, acc_test)

# %%