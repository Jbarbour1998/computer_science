# %% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List

# %% MODEL CONFIGURATION
class ANNConfig:
    def __init__(self, input_size=None, activation="relu"):
        self.input_size = input_size  # Set dynamically based on data
        self.hidden_size = 7
        self.output_size = 1
        self.learning_rate = 0.05
        self.batch_size = 32
        self.epochs = 300
        self.random_seed = 42
        self.activation = activation  # 'relu' or 'sigmoid'

# %% DATA PREPROCESSING
def load_and_preprocess_data(filepath: str):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1).astype(float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, X.shape[1]  # Return input_size too

# %% MODEL ARCHITECTURE
class NeuralNetwork:
    def __init__(self, config: ANNConfig):
        self.config = config
        if self.config.input_size is None:
            raise ValueError("❌ ERROR: input_size must be set before initializing the model!")
        self.parameters = self._initialize_parameters()
    
    def _initialize_parameters(self):
        np.random.seed(self.config.random_seed)
        w1_std = np.sqrt(2.0 / (self.config.input_size + self.config.hidden_size))
        w2_std = np.sqrt(2.0 / (self.config.hidden_size + self.config.output_size))

        return {
            'W1': np.random.normal(0, w1_std, (self.config.hidden_size, self.config.input_size)),
            'b1': np.zeros((self.config.hidden_size, 1)),
            'W2': np.random.normal(0, w2_std, (self.config.output_size, self.config.hidden_size)),
            'b2': np.zeros((self.config.output_size, 1))
        }
    
    @staticmethod
    def relu(Z): return np.maximum(0, Z)
    @staticmethod
    def relu_derivative(Z): return (Z > 0).astype(float)
    @staticmethod
    def sigmoid(Z): return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
    @staticmethod
    def sigmoid_derivative(A): return A * (1 - A)

    def forward_propagation(self, X):
        Z1 = np.dot(self.parameters['W1'], X.T) + self.parameters['b1']
        A1 = self.relu(Z1) if self.config.activation == "relu" else self.sigmoid(Z1)

        Z2 = np.dot(self.parameters['W2'], A1) + self.parameters['b2']
        A2 = self.sigmoid(Z2)  # Always use Sigmoid in output layer

        return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    
    def compute_loss(self, A2, Y):
        epsilon = 1e-15
        return -np.mean(Y.T * np.log(A2 + epsilon) + (1 - Y.T) * np.log(1 - A2 + epsilon))
    
    def backward_propagation(self, X, Y, cache):
        m = X.shape[0]

        dZ2 = cache['A2'] - Y.T
        dW2 = (1/m) * np.dot(dZ2, cache['A1'].T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        if self.config.activation == "relu":
            dZ1 = np.dot(self.parameters['W2'].T, dZ2) * self.relu_derivative(cache['Z1'])
        else:
            dZ1 = np.dot(self.parameters['W2'].T, dZ2) * self.sigmoid_derivative(cache['A1'])

        dW1 = (1/m) * np.dot(dZ1, X)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def train(self, X_train, Y_train):
        m = X_train.shape[0]

        for epoch in range(self.config.epochs):
            permutation = np.random.permutation(m)
            X_train_shuffled = X_train[permutation]
            Y_train_shuffled = Y_train[permutation]

            for i in range(0, m, self.config.batch_size):
                batch_X = X_train_shuffled[i:i + self.config.batch_size]
                batch_Y = Y_train_shuffled[i:i + self.config.batch_size]
                
                cache = self.forward_propagation(batch_X)
                gradients = self.backward_propagation(batch_X, batch_Y, cache)

                for param in ['W1', 'b1', 'W2', 'b2']:
                    self.parameters[param] -= self.config.learning_rate * gradients[f'd{param}']

        return self.evaluate(X_train, Y_train)
    
    def evaluate(self, X, Y):
        cache = self.forward_propagation(X)
        predictions = (cache['A2'] > 0.5).astype(int).T
        accuracy = np.mean(predictions == Y)
        loss = self.compute_loss(cache['A2'], Y)
        return {'accuracy': accuracy, 'loss': loss}

# %% CROSS-VALIDATION
def cross_validation(X, y, input_size, activation="relu", k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    accuracies, losses = [], []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        config = ANNConfig(input_size=input_size, activation=activation)
        model = NeuralNetwork(config)
        metrics = model.train(X_train, Y_train)
        
        accuracies.append(metrics['accuracy'])
        losses.append(metrics['loss'])

        print(f"Fold {fold} - {activation.upper()} Accuracy: {metrics['accuracy']:.4f}")

    return accuracies, losses


# %% PLOT RESULTS
def plot_results(results):
    activations = ["relu", "sigmoid"]
    metrics = ["accuracy", "loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, metric in enumerate(metrics):
        for activation in activations:
            values = results[activation][metric]
            axes[i].plot(range(1, len(values) + 1), values, label=f"{activation.capitalize()} Activation")
        
        axes[i].set_xlabel("Fold")
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f"Cross-Validation {metric.capitalize()}")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

# %% MAIN FUNCTION
def main():
    X, y, input_size = load_and_preprocess_data("heart-disease.csv")

    # Run cross-validation for both activation functions
    results = {}
    for activation in ["relu", "sigmoid"]:
        accuracies, losses = cross_validation(X, y, input_size, activation)
        results[activation] = {"accuracy": accuracies, "loss": losses}

    # Plot performance comparison
    plot_results(results)
    results["relu"]["accuracy"]
    
    


if __name__ == "__main__":
    main()
# %%