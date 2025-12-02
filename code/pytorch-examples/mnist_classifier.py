"""
PyTorch MNIST Classifier
========================

This module demonstrates training a neural network on MNIST using PyTorch.

Topics covered:
- PyTorch Dataset and DataLoader
- Model definition
- Training loop
- Evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class MNISTClassifier(nn.Module):
    """
    Simple feedforward neural network for MNIST classification.
    
    Architecture:
        Input (784) -> FC(256) -> ReLU -> Dropout -> 
        FC(128) -> ReLU -> Dropout -> FC(10)
    """
    
    def __init__(self, dropout_rate=0.2):
        super(MNISTClassifier, self).__init__()
        
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Flatten image: (batch, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        
        # First hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (logits)
        x = self.fc3(x)
        
        return x


class CNNClassifier(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.
    
    Architecture:
        Conv2d -> ReLU -> MaxPool ->
        Conv2d -> ReLU -> MaxPool ->
        FC -> ReLU -> FC
    """
    
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Add channel dimension if needed: (batch, 28, 28) -> (batch, 1, 28, 28)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Conv block 1: (batch, 1, 28, 28) -> (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2: (batch, 32, 14, 14) -> (batch, 64, 7, 7)
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten: (batch, 64, 7, 7) -> (batch, 64*7*7)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def generate_fake_mnist(n_samples=1000):
    """Generate fake MNIST-like data for demonstration."""
    # Random images (28x28)
    X = np.random.rand(n_samples, 28, 28).astype(np.float32)
    # Random labels (0-9)
    y = np.random.randint(0, 10, n_samples)
    return X, y


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    """Create PyTorch DataLoaders."""
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    
    return test_loss, test_acc


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    """Full training loop."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    return history


def demo_pytorch_mnist():
    """Demonstrate PyTorch training on MNIST-like data."""
    
    print("=" * 70)
    print("PYTORCH MNIST CLASSIFIER DEMO")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Generate fake MNIST data (in real usage, use torchvision.datasets.MNIST)
    print("\n1. Generating fake MNIST-like data...")
    X_train, y_train = generate_fake_mnist(n_samples=5000)
    X_test, y_test = generate_fake_mnist(n_samples=1000)
    
    print(f"   Training: {X_train.shape}, {y_train.shape}")
    print(f"   Test: {X_test.shape}, {y_test.shape}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, batch_size=64
    )
    
    # ========== MLP Model ==========
    print("\n" + "=" * 70)
    print("TRAINING MLP MODEL")
    print("=" * 70)
    
    mlp = MNISTClassifier(dropout_rate=0.2)
    print(f"\nModel Architecture:\n{mlp}")
    
    # Count parameters
    total_params = sum(p.numel() for p in mlp.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Train
    print("\nTraining...")
    mlp_history = train_model(mlp, train_loader, test_loader, epochs=5, lr=0.001, device=device)
    
    # ========== CNN Model ==========
    print("\n" + "=" * 70)
    print("TRAINING CNN MODEL")
    print("=" * 70)
    
    cnn = CNNClassifier()
    print(f"\nModel Architecture:\n{cnn}")
    
    total_params = sum(p.numel() for p in cnn.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Train
    print("\nTraining...")
    cnn_history = train_model(cnn, train_loader, test_loader, epochs=5, lr=0.001, device=device)
    
    # ========== Model Comparison ==========
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Model':<15} {'Final Train Acc':<18} {'Final Test Acc':<18}")
    print("-" * 50)
    print(f"{'MLP':<15} {mlp_history['train_acc'][-1]:<18.4f} {mlp_history['test_acc'][-1]:<18.4f}")
    print(f"{'CNN':<15} {cnn_history['train_acc'][-1]:<18.4f} {cnn_history['test_acc'][-1]:<18.4f}")
    
    # Sample inference
    print("\n" + "=" * 70)
    print("SAMPLE INFERENCE")
    print("=" * 70)
    
    # Get a batch
    sample_data, sample_labels = next(iter(test_loader))
    sample_data = sample_data[:5].to(device)
    sample_labels = sample_labels[:5]
    
    # Predictions
    mlp.eval()
    cnn.eval()
    
    with torch.no_grad():
        mlp_pred = mlp(sample_data).argmax(dim=1).cpu()
        cnn_pred = cnn(sample_data).argmax(dim=1).cpu()
    
    print(f"\n{'True':<10} {'MLP Pred':<12} {'CNN Pred':<12}")
    print("-" * 34)
    for true, mlp_p, cnn_p in zip(sample_labels, mlp_pred, cnn_pred):
        print(f"{true.item():<10} {mlp_p.item():<12} {cnn_p.item():<12}")
    
    return mlp, cnn


if __name__ == "__main__":
    demo_pytorch_mnist()
