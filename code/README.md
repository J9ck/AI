# 💻 Code Examples

Welcome to the code section! This directory contains practical implementations and examples for various AI/ML concepts.

## 📁 Directory Structure

| Directory | Description | Key Files |
|-----------|-------------|-----------|
| 🔢 **ml-basics/** | Fundamental ML algorithms | Linear regression, Decision trees, Random forest |
| 🧠 **neural-networks/** | Neural network implementations | MLP from scratch, Backpropagation |
| 🔥 **pytorch-examples/** | PyTorch tutorials | Training loops, Custom datasets, Models |
| 🤗 **huggingface-examples/** | Transformers library usage | Fine-tuning, Inference, Pipelines |
| 💬 **prompt-engineering/** | LLM prompting techniques | Few-shot, Chain-of-thought, Structured output |
| 📊 **data-preprocessing/** | Data preparation utilities | Cleaning, Feature engineering, Scaling |

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn torch transformers
```

### Running Examples

```bash
# ML Basics
python ml-basics/linear_regression.py

# Neural Networks
python neural-networks/mlp_from_scratch.py

# PyTorch
python pytorch-examples/mnist_classifier.py

# Hugging Face
python huggingface-examples/sentiment_analysis.py
```

## 📚 Learning Path

```
START HERE
    │
    ▼
┌────────────────┐
│   ML Basics    │  Understand fundamental algorithms
└───────┬────────┘
        │
        ▼
┌────────────────┐
│Neural Networks │  Build NNs from scratch
└───────┬────────┘
        │
        ▼
┌────────────────┐
│    PyTorch     │  Use modern deep learning framework
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Hugging Face   │  Work with pre-trained transformers
└───────┬────────┘
        │
        ▼
┌────────────────┐
│Prompt Eng.     │  Master LLM interactions
└────────────────┘
```

## 🎯 Best Practices

1. **Read the comments** - Each file is thoroughly documented
2. **Run the code** - Experiment with parameters and data
3. **Break things** - Understanding errors is learning
4. **Extend examples** - Add your own modifications

---

🌐 [Back to Main Repository](../README.md) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
# 💻 Code Examples & Implementations

> Practical code examples for Machine Learning and Deep Learning concepts.

[← Back to Main](../README.md)

---

## 📋 Table of Contents

- [Basic ML Models](#-basic-ml-models)
- [Neural Network from Scratch](#-neural-network-from-scratch)
- [PyTorch Basics](#-pytorch-basics)
- [TensorFlow Basics](#-tensorflow-basics)
- [Hugging Face Transformers](#-hugging-face-transformers)
- [Prompt Engineering Examples](#-prompt-engineering-examples)
- [Data Preprocessing](#-data-preprocessing)
- [Model Evaluation](#-model-evaluation)

---

## 🎯 Basic ML Models

### Linear Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
```

### Logistic Regression (Classification)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Binary classification (classes 0 and 1 only)
mask = y < 2
X, y = X[mask], y[mask]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train model
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True)
plt.title("Decision Tree for Iris Classification")
plt.tight_layout()
plt.savefig("decision_tree.png")
plt.show()

# Feature importance
for name, importance in zip(iris.feature_names, clf.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Evaluate
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

### K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X, y_true = make_blobs(
    n_samples=300, 
    centers=4, 
    cluster_std=0.60, 
    random_state=42
)

# Train K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title("True Clusters")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1],
            marker='x', s=200, linewidths=3, color='red')
plt.title("K-Means Clusters")

plt.tight_layout()
plt.show()

# Elbow method to find optimal K
inertias = []
K_range = range(1, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline with scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")
```

---

## 🧠 Neural Network from Scratch

### Simple Neural Network with NumPy

```python
import numpy as np

class NeuralNetwork:
    """
    A simple 2-layer neural network from scratch.
    Architecture: Input -> Hidden (ReLU) -> Output (Sigmoid)
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def train(self, X, y, epochs=1000, verbose=True):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# Example usage
if __name__ == "__main__":
    # XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    losses = nn.train(X, y, epochs=1000)
    
    # Predictions
    predictions = nn.predict(X)
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Predicted: {predictions[i][0]}, Actual: {y[i][0]}")
```

---

## 🔥 PyTorch Basics

### Simple Neural Network in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Create sample data
X_train = torch.randn(1000, 10)
y_train = torch.randint(0, 3, (1000,))

# Create DataLoader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = SimpleNN(input_size=10, hidden_size=64, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), 'model.pth')
print("Model saved!")
```

### CNN for Image Classification (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total

# Evaluation
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# Train for 5 epochs
for epoch in range(5):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
```

---

## 📊 TensorFlow Basics

### Simple Model with Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate sample data
X_train = np.random.randn(1000, 20).astype(np.float32)
y_train = (np.sum(X_train[:, :5], axis=1) > 0).astype(np.float32)

X_test = np.random.randn(200, 20).astype(np.float32)
y_test = (np.sum(X_test[:, :5], axis=1) > 0).astype(np.float32)

# Build model using Sequential API
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")
```

### Custom Model with Functional API

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# Define model using Functional API
def create_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # First branch
    x1 = layers.Dense(64, activation='relu')(inputs)
    x1 = layers.Dropout(0.3)(x1)
    
    # Second branch
    x2 = layers.Dense(64, activation='relu')(inputs)
    x2 = layers.Dropout(0.3)(x2)
    
    # Merge branches
    merged = layers.concatenate([x1, x2])
    
    # Output layers
    x = layers.Dense(32, activation='relu')(merged)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create model
model = create_model(input_shape=(20,), num_classes=3)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Custom training with callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

# model.fit(..., callbacks=callbacks)
```

### CNN in TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def create_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Convolutional blocks
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

# Usage for CIFAR-10
model = create_cnn(input_shape=(32, 32, 3), num_classes=10)
model.summary()
```

---

## 🤗 Hugging Face Transformers

### Text Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
)

# Example inference
texts = [
    "I love this product! It's amazing!",
    "Terrible experience. Very disappointed."
]

# Tokenize
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

for text, pred in zip(texts, predictions):
    label = "Positive" if pred[1] > pred[0] else "Negative"
    confidence = pred.max().item()
    print(f"Text: {text}")
    print(f"Prediction: {label} ({confidence:.2%})\n")
```

### Text Generation with GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Generate text
prompt = "Artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate with different strategies
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### Named Entity Recognition

```python
from transformers import pipeline

# Load NER pipeline
ner = pipeline("ner", aggregation_strategy="simple")

# Example text
text = """
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne 
in Cupertino, California on April 1, 1976. The company is now valued 
at over $2 trillion and employs more than 160,000 people worldwide.
"""

# Get entities
entities = ner(text)

print("Named Entities Found:")
print("-" * 50)
for entity in entities:
    print(f"{entity['entity_group']:10} | {entity['word']:20} | Score: {entity['score']:.4f}")
```

### Question Answering

```python
from transformers import pipeline

# Load QA pipeline
qa_pipeline = pipeline("question-answering")

# Context and questions
context = """
Machine learning is a subset of artificial intelligence (AI) that provides 
systems the ability to automatically learn and improve from experience without 
being explicitly programmed. Machine learning focuses on the development of 
computer programs that can access data and use it to learn for themselves.
The process begins with observations or data, such as examples, direct 
experience, or instruction, in order to look for patterns in data and make 
better decisions in the future.
"""

questions = [
    "What is machine learning?",
    "What does machine learning focus on?",
    "How does the process begin?"
]

print("Question Answering Results:")
print("=" * 60)
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    print(f"   Score: {result['score']:.4f}")
```

### Embeddings and Semantic Search

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load model for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    """Get embedding for a text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (torch.sum(token_embeddings * input_mask_expanded, 1) / 
            torch.clamp(input_mask_expanded.sum(1), min=1e-9)).numpy()

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example: Semantic search
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "Deep learning uses neural networks with many layers"
]

query = "What is AI and ML?"

# Get embeddings
doc_embeddings = [get_embedding(doc)[0] for doc in documents]
query_embedding = get_embedding(query)[0]

# Compute similarities
similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]

# Rank documents
ranked = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

print(f"Query: {query}\n")
print("Ranked Results:")
for doc, score in ranked:
    print(f"  {score:.4f}: {doc}")
```

---

## 💡 Prompt Engineering Examples

### Zero-Shot Classification

```python
from transformers import pipeline

# Zero-shot classification
classifier = pipeline("zero-shot-classification")

text = "The new iPhone 15 features a titanium design and improved camera system."

candidate_labels = ["technology", "sports", "politics", "entertainment", "science"]

result = classifier(text, candidate_labels)

print(f"Text: {text}\n")
print("Classification Results:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label}: {score:.4f}")
```

### Structured Prompting

```python
# Examples of different prompting strategies

# 1. Few-Shot Prompting
few_shot_prompt = """
Classify the sentiment of the review as Positive, Negative, or Neutral.

Review: "This product exceeded my expectations!"
Sentiment: Positive

Review: "Waste of money. Don't buy this."
Sentiment: Negative

Review: "It's okay, nothing special."
Sentiment: Neutral

Review: "Absolutely fantastic! Best purchase I've made all year."
Sentiment:"""

# 2. Chain-of-Thought Prompting
cot_prompt = """
Question: A store has 45 apples. If they sell 23 apples and then receive a shipment of 18 more, how many apples do they have?

Let me think through this step by step:
1. Starting apples: 45
2. After selling 23: 45 - 23 = 22 apples
3. After receiving 18 more: 22 + 18 = 40 apples

Answer: 40 apples

Question: A farmer has 156 chickens. If 47 chickens are sold and 32 new chickens are born, how many chickens does the farmer have?

Let me think through this step by step:"""

# 3. Instruction Following
instruction_prompt = """
You are a helpful assistant that extracts key information from text.

Extract the following from the text below:
- Person names
- Organizations  
- Locations
- Dates

Text: "On March 15, 2024, Elon Musk announced that Tesla would open a new factory in Austin, Texas."

Extracted Information:
- Person names: Elon Musk
- Organizations: Tesla
- Locations: Austin, Texas
- Dates: March 15, 2024
"""

# 4. Role-Based Prompting
role_prompt = """
You are an expert Python programmer with 20 years of experience. 
Your task is to review code and suggest improvements.

Please review this code and suggest improvements:

```python
def calc(x):
    r = []
    for i in range(len(x)):
        r.append(x[i] * 2)
    return r
```

Code Review:"""

print("Various Prompting Templates Created")
print("Use these with your preferred LLM API")
```

---

## 🔧 Data Preprocessing

### Complete Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Sample data
data = {
    'age': [25, 30, np.nan, 45, 35, 28, np.nan, 50],
    'salary': [50000, 60000, 75000, np.nan, 80000, 55000, 90000, np.nan],
    'department': ['IT', 'HR', 'IT', 'Sales', 'HR', None, 'IT', 'Sales'],
    'experience': ['junior', 'senior', 'mid', 'senior', 'mid', 'junior', 'senior', 'mid'],
    'performance': [3.5, 4.2, 3.8, 4.5, 4.0, 3.2, 4.8, 4.1]
}
df = pd.DataFrame(data)

print("Original Data:")
print(df)
print("\n" + "="*50 + "\n")

# Identify column types
numeric_features = ['age', 'salary']
categorical_features = ['department', 'experience']

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split data
X = df.drop('performance', axis=1)
y = df['performance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names
cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names)

print("Processed Training Data:")
print(pd.DataFrame(X_train_processed, columns=feature_names))
```

### Text Preprocessing

```python
import re
from collections import Counter

def preprocess_text(text, 
                    lowercase=True,
                    remove_punctuation=True,
                    remove_numbers=True,
                    remove_stopwords=True,
                    lemmatize=False):
    """
    Comprehensive text preprocessing function.
    """
    # Basic stopwords (expand as needed)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'it', 'its',
                 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they'}
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    
    return ' '.join(tokens)


# Example usage
sample_text = """
Hello! This is a sample text with some URLs like https://example.com 
and email@test.com. It also has numbers like 12345 and some HTML <b>tags</b>.
The quick brown fox jumps over the lazy dog!!!
"""

processed = preprocess_text(sample_text)
print(f"Original: {sample_text}")
print(f"\nProcessed: {processed}")

# Vocabulary building
def build_vocabulary(texts, min_freq=1, max_vocab_size=None):
    """Build vocabulary from list of texts."""
    word_counts = Counter()
    for text in texts:
        tokens = text.lower().split()
        word_counts.update(tokens)
    
    # Filter by frequency
    vocab = {word: count for word, count in word_counts.items() if count >= min_freq}
    
    # Sort by frequency
    vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))
    
    # Limit size
    if max_vocab_size:
        vocab = dict(list(vocab.items())[:max_vocab_size])
    
    # Create word to index mapping
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word in vocab:
        word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx, vocab

# Example
texts = [
    "machine learning is great",
    "deep learning is amazing",
    "machine learning and deep learning"
]
word_to_idx, vocab = build_vocabulary(texts)
print(f"\nVocabulary: {vocab}")
print(f"Word to Index: {word_to_idx}")
```

---

## 📊 Model Evaluation

### Classification Metrics

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt

# Sample predictions
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
y_prob = np.array([0.1, 0.7, 0.8, 0.9, 0.2, 0.4, 0.3, 0.85, 0.75, 0.6])

# Basic metrics
print("Classification Metrics")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_true, y_prob):.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

# Plot ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
axes[0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_true, y_prob):.3f})')
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()

# PR Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
axes[1].plot(recall, precision, label=f'PR (AP = {ap:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()

plt.tight_layout()
plt.savefig('evaluation_curves.png')
plt.show()
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)
import numpy as np

# Sample data
y_true = np.array([3.0, 2.5, 4.0, 5.5, 3.5, 4.5, 5.0, 2.0])
y_pred = np.array([2.8, 2.7, 3.8, 5.2, 3.8, 4.3, 4.7, 2.3])

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print("Regression Metrics")
print("=" * 50)
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.4%}")

# Residual Analysis
residuals = y_true - y_pred
print(f"\nResidual Statistics:")
print(f"  Mean:   {np.mean(residuals):.4f}")
print(f"  Std:    {np.std(residuals):.4f}")
print(f"  Min:    {np.min(residuals):.4f}")
print(f"  Max:    {np.max(residuals):.4f}")
```

### Cross-Validation

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate, 
    KFold, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

# Generate sample data
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_classes=2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Simple cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Simple Cross-Validation")
print(f"  Scores: {scores}")
print(f"  Mean:   {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Multiple metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

print("\nMultiple Metrics Cross-Validation:")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"  {metric:12}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Stratified K-Fold (maintains class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    fold_scores.append(score)
    print(f"Fold {fold + 1}: {score:.4f}")

print(f"\nStratified K-Fold Mean: {np.mean(fold_scores):.4f}")
```

---

<div align="center">

## 📚 Continue Learning

| Section | Link |
|---------|------|
| 📚 Notes | [Browse Notes →](../notes/README.md) |
| 🔗 Resources | [Browse Resources →](../resources/README.md) |
| 📋 Cheatsheets | [Browse Cheatsheets →](../cheatsheets/README.md) |
| 📖 Glossary | [Browse Glossary →](../glossary/README.md) |

---

[← Back to Main](../README.md)

</div>
