"""
Multi-Layer Perceptron (MLP) from Scratch
==========================================

This module implements a neural network from scratch using only NumPy.

Topics covered:
- Forward propagation
- Backpropagation
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (Cross-Entropy)
- Mini-batch gradient descent
"""

import numpy as np
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    """
    Multi-Layer Perceptron implemented from scratch.
    
    Architecture: Input -> Hidden Layers -> Output
    
    Parameters
    ----------
    layer_sizes : list
        Number of neurons in each layer, including input and output
        Example: [2, 64, 32, 3] for 2 inputs, 2 hidden layers, 3 outputs
    learning_rate : float
        Step size for gradient descent
    activation : str
        Activation function for hidden layers ('relu' or 'sigmoid')
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, activation='relu'):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.activation_name = activation
        self.n_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU, Xavier for sigmoid
            if activation == 'relu':
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])
            
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(W)
            self.biases.append(b)
        
        self.loss_history = []
    
    # ==================== Activation Functions ====================
    
    def relu(self, z):
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """Sigmoid activation: 1 / (1 + exp(-z))"""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z))"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def softmax(self, z):
        """Softmax activation for output layer"""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def activation(self, z):
        """Apply activation function"""
        if self.activation_name == 'relu':
            return self.relu(z)
        return self.sigmoid(z)
    
    def activation_derivative(self, z):
        """Get derivative of activation function"""
        if self.activation_name == 'relu':
            return self.relu_derivative(z)
        return self.sigmoid_derivative(z)
    
    # ==================== Forward Propagation ====================
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        output : np.ndarray
            Network predictions
        cache : dict
            Intermediate values for backpropagation
        """
        cache = {'A': [X], 'Z': []}
        
        A = X
        for i in range(self.n_layers - 2):
            # Linear transformation
            Z = A @ self.weights[i] + self.biases[i]
            # Activation (hidden layers)
            A = self.activation(Z)
            
            cache['Z'].append(Z)
            cache['A'].append(A)
        
        # Output layer (softmax for classification)
        Z = A @ self.weights[-1] + self.biases[-1]
        output = self.softmax(Z)
        
        cache['Z'].append(Z)
        cache['A'].append(output)
        
        return output, cache
    
    # ==================== Backward Propagation ====================
    
    def backward(self, y, cache):
        """
        Backward pass to compute gradients.
        
        Parameters
        ----------
        y : np.ndarray, shape (n_samples, n_classes)
            One-hot encoded true labels
        cache : dict
            Cached values from forward pass
            
        Returns
        -------
        gradients : dict
            Gradients for weights and biases
        """
        m = y.shape[0]  # Number of samples
        gradients = {'dW': [], 'db': []}
        
        # Output layer gradient (softmax + cross-entropy)
        # dL/dZ = A - Y (simplified gradient for softmax + CE)
        dZ = cache['A'][-1] - y
        
        # Backpropagate through layers
        for i in range(self.n_layers - 2, -1, -1):
            A_prev = cache['A'][i]
            
            # Compute gradients
            dW = (1/m) * (A_prev.T @ dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            gradients['dW'].insert(0, dW)
            gradients['db'].insert(0, db)
            
            # Propagate gradient to previous layer (except for input layer)
            if i > 0:
                dA = dZ @ self.weights[i].T
                dZ = dA * self.activation_derivative(cache['Z'][i-1])
        
        return gradients
    
    # ==================== Loss Function ====================
    
    def cross_entropy_loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss.
        
        L = -1/m * sum(y * log(y_pred))
        """
        m = y_true.shape[0]
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    
    # ==================== Training ====================
    
    def fit(self, X, y, epochs=1000, batch_size=32, verbose=True):
        """
        Train the neural network.
        
        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            True labels (will be one-hot encoded)
        epochs : int
            Number of training epochs
        batch_size : int
            Mini-batch size
        verbose : bool
            Print progress
        """
        # One-hot encode labels
        n_classes = len(np.unique(y))
        y_onehot = np.eye(n_classes)[y]
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                output, cache = self.forward(X_batch)
                
                # Compute loss
                loss = self.cross_entropy_loss(output, y_batch)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                gradients = self.backward(y_batch, cache)
                
                # Update weights
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * gradients['dW'][i]
                    self.biases[i] -= self.lr * gradients['db'][i]
            
            epoch_loss /= n_batches
            self.loss_history.append(epoch_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                accuracy = self.score(X, y)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f}")
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        output, _ = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        output, _ = self.forward(X)
        return output
    
    def score(self, X, y):
        """Calculate accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def demo_neural_network():
    """Demonstrate the neural network implementation."""
    
    print("=" * 60)
    print("NEURAL NETWORK FROM SCRATCH DEMO")
    print("=" * 60)
    
    # Generate dataset
    print("\n1. Generating Moon dataset...")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Create and train network
    print("\n2. Creating Neural Network...")
    print("   Architecture: 2 -> 64 -> 32 -> 2")
    
    nn = NeuralNetwork(
        layer_sizes=[2, 64, 32, 2],
        learning_rate=0.1,
        activation='relu'
    )
    
    print("\n3. Training...")
    nn.fit(X_train, y_train, epochs=500, batch_size=32, verbose=True)
    
    # Evaluate
    print("\n4. Evaluation")
    print("-" * 40)
    
    train_acc = nn.score(X_train, y_train)
    test_acc = nn.score(X_test, y_test)
    
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # Sample predictions
    print("\n5. Sample Predictions")
    print("-" * 40)
    
    sample_X = X_test[:5]
    sample_y = y_test[:5]
    predictions = nn.predict(sample_X)
    probabilities = nn.predict_proba(sample_X)
    
    print(f"   {'True':<8} {'Pred':<8} {'Prob Class 0':<15} {'Prob Class 1':<15}")
    for true, pred, proba in zip(sample_y, predictions, probabilities):
        print(f"   {true:<8} {pred:<8} {proba[0]:<15.4f} {proba[1]:<15.4f}")
    
    # Network summary
    print("\n6. Network Summary")
    print("-" * 40)
    
    total_params = 0
    for i, (W, b) in enumerate(zip(nn.weights, nn.biases)):
        layer_params = W.size + b.size
        total_params += layer_params
        print(f"   Layer {i+1}: W shape {W.shape}, b shape {b.shape} -> {layer_params} params")
    print(f"   Total parameters: {total_params}")
    
    return nn


def demo_gradient_check():
    """Verify gradients using numerical gradient checking."""
    
    print("\n" + "=" * 60)
    print("GRADIENT CHECKING")
    print("=" * 60)
    
    # Small network for gradient checking
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, 10)
    y_onehot = np.eye(2)[y]
    
    nn = NeuralNetwork([2, 4, 2], learning_rate=0.1)
    
    # Analytical gradients
    output, cache = nn.forward(X)
    grads = nn.backward(y_onehot, cache)
    
    # Numerical gradients
    epsilon = 1e-5
    numerical_grads = []
    
    for i in range(len(nn.weights)):
        dW_num = np.zeros_like(nn.weights[i])
        
        for j in range(nn.weights[i].shape[0]):
            for k in range(nn.weights[i].shape[1]):
                # w + epsilon
                nn.weights[i][j, k] += epsilon
                output_plus, _ = nn.forward(X)
                loss_plus = nn.cross_entropy_loss(output_plus, y_onehot)
                
                # w - epsilon
                nn.weights[i][j, k] -= 2 * epsilon
                output_minus, _ = nn.forward(X)
                loss_minus = nn.cross_entropy_loss(output_minus, y_onehot)
                
                # Numerical gradient
                dW_num[j, k] = (loss_plus - loss_minus) / (2 * epsilon)
                
                # Reset weight
                nn.weights[i][j, k] += epsilon
        
        numerical_grads.append(dW_num)
    
    # Compare gradients
    for i in range(len(nn.weights)):
        diff = np.abs(grads['dW'][i] - numerical_grads[i])
        max_diff = np.max(diff)
        print(f"Layer {i+1} - Max gradient difference: {max_diff:.10f}")
        
        if max_diff < 1e-5:
            print(f"   ✓ Gradients match!")
        else:
            print(f"   ✗ Gradient mismatch detected")


if __name__ == "__main__":
    demo_neural_network()
    demo_gradient_check()
