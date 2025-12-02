# 🔥 PyTorch Cheatsheet

Quick reference for PyTorch deep learning.

## Tensors

```python
import torch

# Creation
x = torch.tensor([1, 2, 3])             # From list
x = torch.zeros(3, 4)                    # Zeros
x = torch.ones(3, 4)                     # Ones
x = torch.randn(3, 4)                    # Random normal
x = torch.arange(0, 10, 2)               # Range
x = torch.linspace(0, 1, 5)              # Linear space
x = torch.eye(3)                         # Identity

# From NumPy
x = torch.from_numpy(numpy_array)
numpy_array = x.numpy()

# Properties
x.shape                                  # Dimensions
x.dtype                                  # Data type
x.device                                 # CPU or GPU
x.requires_grad                          # Gradient tracking

# Operations
x + y                                    # Addition
x * y                                    # Element-wise mult
x @ y                                    # Matrix mult
torch.matmul(x, y)                       # Matrix mult
x.sum(), x.mean(), x.std()               # Reductions
x.max(), x.min()                         # Max/min
x.argmax(), x.argmin()                   # Index of max/min

# Reshaping
x.view(2, 6)                             # Reshape (contiguous)
x.reshape(2, 6)                          # Reshape (any)
x.unsqueeze(0)                           # Add dimension
x.squeeze()                              # Remove size-1 dims
x.permute(1, 0, 2)                       # Reorder dimensions
x.flatten()                              # Flatten

# GPU
x = x.to('cuda')                         # Move to GPU
x = x.to('cpu')                          # Move to CPU
x = x.cuda()                             # Move to GPU
```

## Neural Network Modules

```python
import torch.nn as nn
import torch.nn.functional as F

# Basic Layers
nn.Linear(in_features, out_features)     # Fully connected
nn.Conv2d(in_channels, out_channels, kernel_size)  # 2D convolution
nn.MaxPool2d(kernel_size)                # Max pooling
nn.BatchNorm2d(num_features)             # Batch normalization
nn.Dropout(p=0.5)                        # Dropout
nn.LSTM(input_size, hidden_size)         # LSTM
nn.Embedding(num_embeddings, embedding_dim)  # Embedding

# Activations
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)
nn.LeakyReLU(0.1)
nn.GELU()

# Functional versions (no state)
F.relu(x)
F.softmax(x, dim=1)
F.cross_entropy(output, target)
```

## Model Definition

```python
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Using nn.Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)
```

## Training Loop

```python
# Setup
model = MyModel(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
```

## Data Loading

```python
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

# DataLoader
dataset = MyDataset(X, y, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # For faster GPU transfer
)
```

## Optimizers & Schedulers

```python
# Optimizers
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Learning Rate Schedulers
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Use in training loop
for epoch in range(num_epochs):
    train_one_epoch()
    scheduler.step()  # or scheduler.step(val_loss) for ReduceLROnPlateau
```

## Saving & Loading

```python
# Save entire model
torch.save(model, 'model.pt')
model = torch.load('model.pt')

# Save only weights (recommended)
torch.save(model.state_dict(), 'model_weights.pt')
model.load_state_dict(torch.load('model_weights.pt'))

# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## CNN Architecture Template

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

## Common Patterns

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Freeze layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(data)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Move model to DataParallel
model = nn.DataParallel(model)
```

---

🌐 [Back to Cheatsheets](.) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
