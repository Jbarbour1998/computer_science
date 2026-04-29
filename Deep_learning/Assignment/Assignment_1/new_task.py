# %% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from typing import Dict, Tuple, List
from sklearn.metrics import roc_curve, auc

# %% MODEL CONFIGURATION
class ANNConfig:
    """Configuration class for ANN hyperparameters"""
    def __init__(self):
        self.input_size: int = None  # Set dynamically based on data
        self.hidden_size: int = 5
        self.output_size: int = 1
        self.learning_rate: float = 0.01
        self.batch_size: int = 128
        self.epochs: int = 500
        self.random_seed: int = 42

# %% DATA PREPROCESSING
def load_and_preprocess_data(filepath: str, config: ANNConfig) -> Tuple[np.ndarray, ...]:
    """
    Load and preprocess the heart disease dataset.
    
    Parameters:
        filepath (str): Path to the dataset
        config (ANNConfig): Model configuration object
    
    Returns:
        tuple: Training and testing data splits (x_train, x_test, y_train, y_test)
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Split features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1).astype(float)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Set input size in config
    config.input_size = X.shape[1]
    
    # Split data
    return train_test_split(X, y, test_size=0.2, random_state=config.random_seed)

# %% MODEL ARCHITECTURE
class NeuralNetwork:
    """Implementation of a simple feedforward neural network"""
    
    def __init__(self, config: ANNConfig):
        """Initialize the neural network with given configuration"""
        self.config = config
        self.parameters = self._initialize_parameters()
        self.training_history = {
            'train_loss': [], 'train_accuracy': [],
            'test_loss': [], 'test_accuracy': [],
            'train_sensitivity': [], 'train_specificity': [],
            'test_sensitivity': [], 'test_specificity': []
        }
    
    def _initialize_parameters(self) -> Dict[str, np.ndarray]:
        """Initialize network parameters with Xavier initialization"""
        np.random.seed(self.config.random_seed)
        
        # Xavier initialization
        w1_std = np.sqrt(2.0 / (self.config.input_size + self.config.hidden_size))
        w2_std = np.sqrt(2.0 / (self.config.hidden_size + self.config.output_size))
        
        return {
            'W1': np.random.normal(0, w1_std, (self.config.hidden_size, self.config.input_size)),
            'b1': np.zeros((self.config.hidden_size, 1)),
            'W2': np.random.normal(0, w2_std, (self.config.output_size, self.config.hidden_size)),
            'b2': np.zeros((self.config.output_size, 1))
        }
    
    @staticmethod
    def sigmoid(Z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Clip to avoid overflow
    
    @staticmethod
    def sigmoid_derivative(A: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid activation"""
        return A * (1 - A)
    
    def forward_propagation(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward propagation step
        
        Parameters:
            X (np.ndarray): Input features
            
        Returns:
            dict: Cache of intermediate values
        """
        # Hidden layer
        Z1 = np.dot(self.parameters['W1'], X.T) + self.parameters['b1']
        A1 = self.sigmoid(Z1)
        
        # Output layer
        Z2 = np.dot(self.parameters['W2'], A1) + self.parameters['b2']
        A2 = self.sigmoid(Z2)
        
        return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    
    def compute_loss(self, A2: np.ndarray, Y: np.ndarray, epsilon: float = 1e-15) -> float:
        """Compute binary cross-entropy loss"""
        m = Y.shape[0]
        loss = -np.mean(Y.T * np.log(A2 + epsilon) + 
                       (1 - Y.T) * np.log(1 - A2 + epsilon))
        return loss
    
    def backward_propagation(self, X: np.ndarray, Y: np.ndarray, 
                           cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Backward propagation step
        
        Returns:
            dict: Gradients for each parameter
        """
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = cache['A2'] - Y.T
        dW2 = (1/m) * np.dot(dZ2, cache['A1'].T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Hidden layer gradients
        dZ1 = np.dot(self.parameters['W2'].T, dZ2) * self.sigmoid_derivative(cache['A1'])
        dW1 = (1/m) * np.dot(dZ1, X)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Returns:
            dict: Dictionary containing various performance metrics
        """
        cache = self.forward_propagation(X)
        predictions = (cache['A2'] > 0.5).astype(int).T
        
        # Calculate metrics
        accuracy = np.mean(predictions == Y)
        
        # Calculate sensitivity (recall)
        true_positives = np.sum((predictions == 1) & (Y == 1))
        actual_positives = np.sum(Y == 1)
        sensitivity = true_positives / actual_positives if actual_positives > 0 else 0
        
        # Calculate specificity
        true_negatives = np.sum((predictions == 0) & (Y == 0))
        actual_negatives = np.sum(Y == 0)
        specificity = true_negatives / actual_negatives if actual_negatives > 0 else 0
        
        # Calculate loss
        loss = self.compute_loss(cache['A2'], Y)
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'loss': loss
        }
    
    def train(self, X_train: np.ndarray, Y_train: np.ndarray, 
              X_test: np.ndarray, Y_test: np.ndarray) -> None:
        """Train the neural network"""
        m = X_train.shape[0]
        
        for epoch in range(self.config.epochs):
            # Mini-batch training
            permutation = np.random.permutation(m)
            X_train_shuffled = X_train[permutation]
            Y_train_shuffled = Y_train[permutation]
            
            for i in range(0, m, self.config.batch_size):
                batch_X = X_train_shuffled[i:i + self.config.batch_size]
                batch_Y = Y_train_shuffled[i:i + self.config.batch_size]
                
                # Forward propagation
                cache = self.forward_propagation(batch_X)
                
                # Backward propagation
                gradients = self.backward_propagation(batch_X, batch_Y, cache)
                
                # Update parameters
                for param in ['W1', 'b1', 'W2', 'b2']:
                    self.parameters[param] -= self.config.learning_rate * gradients[f'd{param}']
            
            # Evaluate and store metrics
            train_metrics = self.evaluate(X_train, Y_train)
            test_metrics = self.evaluate(X_test, Y_test)
            
            # Store training history
            for metric in ['loss', 'accuracy', 'sensitivity', 'specificity']:
                self.training_history[f'train_{metric}'].append(train_metrics[metric])
                self.training_history[f'test_{metric}'].append(test_metrics[metric])
            
            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.config.epochs}")
                print(f"Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
                print(f"Test - Loss: {test_metrics['loss']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}")
    
    def plot_training_history(self) -> None:
        """Plot training metrics history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        ax1.plot(self.training_history['train_loss'], label='Train')
        ax1.plot(self.training_history['test_loss'], label='Test')
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.training_history['train_accuracy'], label='Train')
        ax2.plot(self.training_history['test_accuracy'], label='Test')
        ax2.set_title('Accuracy Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Plot sensitivity
        ax3.plot(self.training_history['train_sensitivity'], label='Train')
        ax3.plot(self.training_history['test_sensitivity'], label='Test')
        ax3.set_title('Sensitivity Over Time')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Sensitivity')
        ax3.legend()
        ax3.grid(True)
        
        # Plot specificity
        ax4.plot(self.training_history['train_specificity'], label='Train')
        ax4.plot(self.training_history['test_specificity'], label='Test')
        ax4.set_title('Specificity Over Time')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Specificity')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray, class_labels=None):
        if class_labels is None:
            class_labels = ["No Disease", "Disease"]
        predictions = (self.forward_propagation(X_test)["A2"] > 0.5).astype(int).T
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
        
    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray):
    
    
        # Get probabilities from forward propagation
        A2_test = self.forward_propagation(X_test)["A2"].T
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, A2_test)
        roc_auc = auc(fpr, tpr)  # Compute AUC
        
        # Plot ROC curve
        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
    def plot_metrics_table(self, interval= 20):
        """Creates a color-coded table to display model performance at every 20th epoch."""
        
        # Convert training history into DataFrame
        metrics_df = pd.DataFrame({
            "Epoch": range(1, len(self.training_history['train_loss']) + 1),
            "Train Loss": self.training_history['train_loss'],
            "Test Loss": self.training_history['test_loss'],
            "Train Accuracy": self.training_history['train_accuracy'],
            "Test Accuracy": self.training_history['test_accuracy'],
            "Train Sensitivity": self.training_history['train_sensitivity'],
            "Test Sensitivity": self.training_history['test_sensitivity'],
            "Train Specificity": self.training_history['train_specificity'],
            "Test Specificity": self.training_history['test_specificity']
        })

        # Select every `interval`th epoch (default = every 20th epoch)
        filtered_metrics_df = metrics_df[metrics_df["Epoch"] % interval == 0]

        # Normalize values between 0 and 1 for color mapping
        norm_df = filtered_metrics_df.iloc[:, 1:].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

        # Create figure & axis
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis('tight')
        ax.axis('off')
        
        # Add title at the top
        ax.text(0.5, 1.2, f"Model Training Metrics (Every {interval} Epochs)", 
                fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)

        # Convert DataFrame to table
        table = ax.table(cellText=filtered_metrics_df.round(4).values, 
                        colLabels=filtered_metrics_df.columns, 
                        cellLoc='center', 
                        loc='center')

        # Apply color gradient to cells
        cmap = plt.cm.managua  # Blue → Red colormap
        for i in range(len(filtered_metrics_df)):
            for j in range(1, len(filtered_metrics_df.columns)):  # Skip the Epoch column
                value = norm_df.iloc[i, j-1]  # Get normalized value (for coloring)
                color = cmap(value)  # Get color from colormap
                #text_color = "white" if value < 0.5 else "black"  # Auto-adjust text color
                
                table[(i+1, j)].set_facecolor(color)
                #table[(i+1, j)].get_text().set_color(text_color)

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([i for i in range(len(filtered_metrics_df.columns))])  # Adjust column width

        plt.show()


# %% TRAINING AND EVALUATION
def main():
    # Initialize configuration
    config = ANNConfig()
    
    # Load and preprocess data
    x_train, x_test, y_train, y_test = load_and_preprocess_data("heart-disease.csv", config)
    
    # Initialize and train model
    model = NeuralNetwork(config)
    model.train(x_train, y_train, x_test, y_test)
    model.plot_confusion_matrix(x_test, y_test)
    model.plot_roc_curve(x_test, y_test)
    model.plot_metrics_table()
    
    # Plot training history
    model.plot_training_history()
    
    # Final evaluation
    final_metrics = model.evaluate(x_test, y_test)
    print("\nFinal Test Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()
# %%
