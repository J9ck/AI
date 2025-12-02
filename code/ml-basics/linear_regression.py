"""
Linear Regression Implementation
================================

This module demonstrates linear regression from scratch and using scikit-learn.

Topics covered:
- Simple linear regression (gradient descent)
- Multiple linear regression
- Scikit-learn implementation
- Evaluation metrics (MSE, R²)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using gradient descent.
    
    The model learns: y = X @ w + b
    
    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent
    n_iterations : int
        Number of gradient descent iterations
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features
        y : np.ndarray, shape (n_samples,)
            Target values
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass: predictions
            y_pred = X @ self.weights + self.bias
            
            # Compute loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # Compute gradients
            # dL/dw = (2/n) * X.T @ (y_pred - y)
            # dL/db = (2/n) * sum(y_pred - y)
            dw = (2 / n_samples) * (X.T @ (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {loss:.6f}")
        
        return self
    
    def predict(self, X):
        """Make predictions on new data."""
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


def generate_sample_data(n_samples=100, n_features=1, noise=10, seed=42):
    """Generate synthetic data for regression."""
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features) * 10
    
    # True relationship: y = 3*x + 7 (for 1D case)
    true_weights = np.array([3] + [0.5] * (n_features - 1))
    y = X @ true_weights + 7 + np.random.randn(n_samples) * noise
    
    return X, y


def demo_linear_regression():
    """Demonstrate linear regression implementations."""
    
    print("=" * 60)
    print("LINEAR REGRESSION DEMO")
    print("=" * 60)
    
    # Generate data
    X, y = generate_sample_data(n_samples=100, n_features=1, noise=5)
    
    # Split data (simple split, no sklearn dependency)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nData shapes:")
    print(f"  Training: X={X_train.shape}, y={y_train.shape}")
    print(f"  Testing:  X={X_test.shape}, y={y_test.shape}")
    
    # ========== From Scratch Implementation ==========
    print("\n" + "-" * 40)
    print("FROM SCRATCH IMPLEMENTATION")
    print("-" * 40)
    
    model_scratch = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
    model_scratch.fit(X_train, y_train)
    
    y_pred_scratch = model_scratch.predict(X_test)
    mse_scratch = mean_squared_error(y_test, y_pred_scratch)
    r2_scratch = model_scratch.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Learned weights: {model_scratch.weights}")
    print(f"  Learned bias: {model_scratch.bias:.4f}")
    print(f"  MSE: {mse_scratch:.4f}")
    print(f"  R² Score: {r2_scratch:.4f}")
    
    # ========== Scikit-learn Implementation ==========
    print("\n" + "-" * 40)
    print("SCIKIT-LEARN IMPLEMENTATION")
    print("-" * 40)
    
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train, y_train)
    
    y_pred_sklearn = model_sklearn.predict(X_test)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = model_sklearn.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Learned weights: {model_sklearn.coef_}")
    print(f"  Learned bias: {model_sklearn.intercept_:.4f}")
    print(f"  MSE: {mse_sklearn:.4f}")
    print(f"  R² Score: {r2_sklearn:.4f}")
    
    # ========== Visualization ==========
    print("\n" + "-" * 40)
    print("Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and fitted line
    ax1 = axes[0]
    ax1.scatter(X_test, y_test, color='blue', alpha=0.6, label='Test Data')
    ax1.plot(X_test, y_pred_scratch, color='red', linewidth=2, 
             label=f'Scratch (R²={r2_scratch:.3f})')
    ax1.plot(X_test, y_pred_sklearn, color='green', linewidth=2, linestyle='--',
             label=f'Sklearn (R²={r2_sklearn:.3f})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title('Linear Regression: Predictions vs Actual')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss curve
    ax2 = axes[1]
    ax2.plot(model_scratch.loss_history, color='red')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Loss Over Iterations')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/linear_regression_demo.png', dpi=100)
    print("Visualization saved to /tmp/linear_regression_demo.png")
    
    return model_scratch, model_sklearn


if __name__ == "__main__":
    demo_linear_regression()
