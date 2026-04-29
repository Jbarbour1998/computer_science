# %%

# Imports for data and training
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %%

df = pd.read_csv("heart-disease.csv")
df

# %% SETTING THE X AND Y
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

scaler = StandardScaler()
x = scaler.fit_transform(x)

# %% SETTING THE TRAIN-TEST SPLIT(80% TRAINING, 20% TESTING)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
def initalize_parameters(input_size, hidden_size, output_size):
    
    np.random.seed(42) # FOR REPRODUCIBITLTY
    
    w1 = np.random.uniform(-1, 1, (hidden_size, input_size)) # weights for hidden layer
    b1 = np.zeros((hidden_size, 1)) # Bias for hidden layer
    
    w2 = np.random.uniform(-1, 1, (output_size, hidden_size)) # weights for output layer
    b2 = np.zeros((output_size, 1)) # Bias for output layer
    
    return w1, b1, w2, b2

# Defining ANN architecture
input_size = x_train.shape[1] # number of features
hidden_size = 8 # Hidden layer size
output_size = 1 # Binary classification layer

w1, b1, w2, b2 = initalize_parameters(input_size, hidden_size, output_size)



# %% FORWARD PASS
def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def forward_propagation(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x.T) + b1 # hidden layer pre-activation
    
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2 # output layer preactivation
    a2 = sigmoid(z2) # output layer activation (probabilty)
    
    return z1, a1, z2, a2


# %% BINARY CROSS-ENTROPY

def compute_loss(a2, y):
    m = y.shape[0] # number of samples
    loss = -np.mean(y.T * np.log(a2) + (1- y.T) * np.log(1-a2))
    return loss

# %% BACK-PROPAGATION

def backpropagation(x, y, w1, b1, w2, b2, z1, a1, z2, a2):
    m = x.shape[0]  # Number of training examples
    
    # Compute gradient of loss w.r.t. output layer (sigmoid derivative)
    dZ2 = a2 - y.T  # Gradient of loss for output layer
    dW2 = (1 / m) * np.dot(dZ2, a1.T)  # Gradient w.r.t W2
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # Gradient w.r.t b2

    # Compute gradient for hidden layer (ReLU derivative)
    dZ1 = np.dot(w2.T, dZ2) * (a1 > 0)  # ReLU Derivative applied on a1
    dW1 = (1 / m) * np.dot(dZ1, x)  # Gradient w.r.t W1
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # Gradient w.r.t b1

    return dW1, db1, dW2, db2

################################################################################
# %%
def train(x_train, y_train, w1, b1, w2, b2, learning_rate=0.01, batch_size=128, epochs=500):
    m = x_train.shape[0]  # Number of samples
    loss_history = []

    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        x_train = x_train[permutation]
        y_train = y_train[permutation]

        for i in range(0, m, batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward propagation
            Z1, A1, Z2, A2 = forward_propagation(x_batch, w1, b1, w2, b2)

            # Compute gradients
            dW1, db1, dW2, db2 = backpropagation(x_batch, y_batch, w1, b1, w2, b2, Z1, A1, Z2, A2)

            # Update weights and biases
            w1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            w2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        # Compute loss after each epoch
        _, _, _, A2_train = forward_propagation(x_train, w1, b1, w2, b2)
        loss = compute_loss(A2_train, y_train)
        loss_history.append(loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return w1, b1, w2, b2, loss_history

# Train the model
w1, b1, w2, b2, loss_history = train(x_train, y_train, w1, b1, w2, b2)

# %% SETTING THE PREDICT FUNCTION

def predict(x, w1, b1, w2, b2):
    
    _, _, _, A2 = forward_propagation(x, w1,b1, w2, b2)
    
    return (A2 > 0.5).astype(int)


y_pred = predict(x_test, w1, b1, w2, b2)
accuracy = np.mean(y_pred.T == y_test)
print(f"Final Test Accuracy: {accuracy:.4f}")


# %% UPDATED VISUAL


import matplotlib.pyplot as plt

def train(
    x_train, 
    y_train, 
    x_test, 
    y_test, 
    w1, 
    b1, 
    w2, 
    b2, 
    learning_rate= 0.01, 
    batch_size=128, 
    epochs=500):
    
    m = x_train.shape[0]  # Number of training samples
    loss_history_train = []  # Training loss
    accuracy_history_train = []  # Training accuracy
    loss_history_test = []  # Test loss
    accuracy_history_test = []  # Test accuracy

    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        x_train = x_train[permutation]
        y_train = y_train[permutation]

        for i in range(0, m, batch_size):
            X_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward propagation
            Z1, A1, Z2, A2 = forward_propagation(X_batch, w1, b1, w2, b2)

            # Compute gradients
            dW1, db1, dW2, db2 = backpropagation(X_batch, y_batch, w1, b1, w2, b2, Z1, A1, Z2, A2)

            # Update weights and biases using mini-batch gradient descent
            w1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            w2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        # Compute loss & accuracy for training set
        _, _, _, A2_train = forward_propagation(x_train, w1, b1, w2, b2)
        loss_train = compute_loss(A2_train, y_train)
        loss_history_train.append(loss_train)
        predictions_train = (A2_train > 0.5).astype(int)
        accuracy_train = np.mean(predictions_train.T == y_train)
        accuracy_history_train.append(accuracy_train)

        # Compute loss & accuracy for test set
        _, _, _, A2_test = forward_propagation(x_test, w1, b1, w2, b2)
        loss_test = compute_loss(A2_test, y_test)
        loss_history_test.append(loss_test)
        predictions_test = (A2_test > 0.5).astype(int)
        accuracy_test = np.mean(predictions_test.T == y_test)
        accuracy_history_test.append(accuracy_test)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss_train:.4f}, Train Acc = {accuracy_train:.4f} | Test Loss = {loss_test:.4f}, Test Acc = {accuracy_test:.4f}")

    # Plot training & test loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history_train, label="Train Loss", color='blue')
    plt.plot(loss_history_test, label="Test Loss", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Model Loss")

    # Plot training & test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history_train, label="Train Accuracy", color='blue')
    plt.plot(accuracy_history_test, label="Test Accuracy", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Model Accuracy")

    plt.show()

    return w1, b1, w2, b2, loss_history_train, accuracy_history_train, loss_history_test, accuracy_history_test

# Retrain with test set tracking
w1, b1, w2, b2, loss_train, acc_train, loss_test, acc_test = train(x_train, y_train, x_test, y_test, w1, b1, w2, b2)





# %%
df_1 = df.iloc[:,:-1]
df_1
# %%
df_2 = df.iloc[:,-1]
df_2
# %%
