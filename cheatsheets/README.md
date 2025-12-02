# 📋 AI/ML Cheatsheets

> Quick reference guides for common AI/ML tasks and concepts.

[← Back to Main](../README.md)

---

## 📋 Table of Contents

- [Python for ML](#-python-for-ml)
- [NumPy Essentials](#-numpy-essentials)
- [Pandas Essentials](#-pandas-essentials)
- [PyTorch vs TensorFlow](#-pytorch-vs-tensorflow)
- [ML Algorithms Comparison](#-ml-algorithms-comparison)
- [Model Evaluation Metrics](#-model-evaluation-metrics)
- [Prompt Engineering Patterns](#-prompt-engineering-patterns)

---

## 🐍 Python for ML

### Essential Imports

```python
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Deep Learning (TensorFlow)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# NLP
from transformers import AutoTokenizer, AutoModel
import spacy

# Utilities
import os
import json
import pickle
from pathlib import Path
from tqdm import tqdm
```

### Virtual Environments

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Save requirements
pip freeze > requirements.txt

# Deactivate
deactivate
```

### List Comprehensions & Generators

```python
# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
matrix = [[i*j for j in range(5)] for i in range(5)]

# Dictionary comprehension
word_lengths = {word: len(word) for word in words}

# Generator expression (memory efficient)
squares_gen = (x**2 for x in range(10))

# Lambda functions
double = lambda x: x * 2
add = lambda x, y: x + y
```

### Common Patterns

```python
# Enumerate for index + value
for idx, value in enumerate(my_list):
    print(f"Index {idx}: {value}")

# Zip for parallel iteration
for a, b in zip(list1, list2):
    print(a, b)

# Dictionary get with default
value = my_dict.get('key', default_value)

# Defaultdict for counting
from collections import defaultdict
counts = defaultdict(int)
for item in items:
    counts[item] += 1

# Counter for frequency
from collections import Counter
freq = Counter(words)
most_common = freq.most_common(10)
```

### Type Hints

```python
from typing import List, Dict, Optional, Tuple, Union, Callable

def process_data(
    data: List[Dict[str, any]],
    threshold: float = 0.5
) -> Tuple[np.ndarray, List[str]]:
    """Process data with type hints."""
    pass

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 10,
    callback: Optional[Callable] = None
) -> Dict[str, float]:
    """Train model with optional callback."""
    pass
```

---

## 🔢 NumPy Essentials

### Array Creation

```python
import numpy as np

# Basic arrays
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))           # 3x4 of zeros
ones = np.ones((2, 3))             # 2x3 of ones
full = np.full((3, 3), 7)          # 3x3 filled with 7
eye = np.eye(4)                    # 4x4 identity matrix
random = np.random.randn(3, 3)     # 3x3 random normal

# Ranges
range_arr = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)   # 5 points from 0 to 1
```

### Array Properties

```python
arr = np.random.randn(3, 4, 5)

arr.shape       # (3, 4, 5)
arr.ndim        # 3
arr.size        # 60
arr.dtype       # float64
```

### Reshaping & Manipulation

```python
# Reshape
arr = np.arange(12)
reshaped = arr.reshape(3, 4)       # 3x4 matrix
flattened = reshaped.flatten()     # Back to 1D

# Transpose
transposed = arr.T
transposed = np.transpose(arr, (1, 0, 2))

# Concatenate
joined = np.concatenate([arr1, arr2], axis=0)
vstacked = np.vstack([arr1, arr2])
hstacked = np.hstack([arr1, arr2])

# Split
parts = np.split(arr, 3, axis=0)
```

### Indexing & Slicing

```python
arr = np.random.randn(5, 4)

arr[0]              # First row
arr[:, 0]           # First column
arr[1:4, 2:4]       # Submatrix
arr[arr > 0]        # Boolean indexing
arr[[0, 2, 4]]      # Fancy indexing

# Conditional assignment
arr[arr < 0] = 0
```

### Operations

```python
# Element-wise
a + b, a - b, a * b, a / b
np.sqrt(arr), np.exp(arr), np.log(arr)
np.sin(arr), np.cos(arr)

# Aggregations
arr.sum(), arr.mean(), arr.std()
arr.min(), arr.max()
arr.sum(axis=0)     # Sum along columns
arr.sum(axis=1)     # Sum along rows
np.argmax(arr)      # Index of max
np.argmin(arr)      # Index of min

# Linear algebra
np.dot(a, b)        # Dot product
a @ b               # Matrix multiplication
np.linalg.inv(arr)  # Inverse
np.linalg.det(arr)  # Determinant
np.linalg.eig(arr)  # Eigenvalues/vectors
```

### Broadcasting Rules

```
Arrays are compatible when:
1. They have the same shape, or
2. One of them has size 1 in that dimension

Examples:
(3, 4) + (4,)    → (3, 4)  ✓
(3, 4) + (3, 1)  → (3, 4)  ✓
(3, 4) + (3,)    → Error   ✗
```

---

## 🐼 Pandas Essentials

### DataFrame Creation

```python
import pandas as pd

# From dictionary
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# From CSV
df = pd.read_csv('data.csv')

# From JSON
df = pd.read_json('data.json')
```

### Selection & Filtering

```python
# Columns
df['column']               # Single column (Series)
df[['col1', 'col2']]       # Multiple columns (DataFrame)

# Rows
df.loc[0]                  # By label
df.iloc[0]                 # By position
df.loc[0:5, 'name']        # Label slicing
df.iloc[0:5, 0:2]          # Position slicing

# Boolean filtering
df[df['age'] > 25]
df[(df['age'] > 25) & (df['city'] == 'NYC')]
df.query('age > 25 and city == "NYC"')
```

### Data Inspection

```python
df.head()          # First 5 rows
df.tail()          # Last 5 rows
df.shape           # (rows, columns)
df.info()          # Data types and memory
df.describe()      # Statistical summary
df.columns         # Column names
df.dtypes          # Data types
df.isnull().sum()  # Missing values count
df.nunique()       # Unique values count
```

### Data Cleaning

```python
# Missing values
df.dropna()                        # Drop rows with NaN
df.fillna(0)                       # Fill NaN with 0
df.fillna(df.mean())               # Fill with mean
df['col'].interpolate()            # Interpolate missing

# Duplicates
df.drop_duplicates()
df.drop_duplicates(subset=['col1'])

# Type conversion
df['col'] = df['col'].astype(int)
df['date'] = pd.to_datetime(df['date'])

# Renaming
df.rename(columns={'old': 'new'})
df.columns = ['a', 'b', 'c']
```

### Grouping & Aggregation

```python
# Group by
df.groupby('category').mean()
df.groupby(['cat1', 'cat2']).sum()

# Multiple aggregations
df.groupby('category').agg({
    'value': ['mean', 'sum', 'count'],
    'other': 'max'
})

# Pivot tables
df.pivot_table(
    values='sales',
    index='product',
    columns='month',
    aggfunc='sum'
)
```

### Merging & Joining

```python
# Merge (SQL-like joins)
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, left_on='key1', right_on='key2')
pd.merge(df1, df2, how='left')   # left, right, outer, inner

# Concatenate
pd.concat([df1, df2], axis=0)    # Stack vertically
pd.concat([df1, df2], axis=1)    # Stack horizontally
```

### Apply Functions

```python
# Apply to column
df['col'].apply(lambda x: x * 2)
df['col'].apply(custom_function)

# Apply to DataFrame
df.apply(lambda row: row['a'] + row['b'], axis=1)

# Vectorized string operations
df['name'].str.lower()
df['name'].str.contains('pattern')
df['name'].str.split('_')
```

---

## ⚡ PyTorch vs TensorFlow

### Quick Comparison

| Aspect | PyTorch | TensorFlow |
|--------|---------|------------|
| **Execution** | Eager by default | Eager (TF2) / Graph |
| **API Style** | Pythonic, intuitive | Keras high-level |
| **Debugging** | Easy (Python debugger) | TF debugger required |
| **Best For** | Research, flexibility | Production, mobile |
| **Ecosystem** | HuggingFace, fast.ai | TFX, TF Lite, TF.js |
| **Deployment** | TorchServe, ONNX | TF Serving, TF Lite |

### Tensor Creation

```python
# PyTorch                          # TensorFlow
import torch                        import tensorflow as tf

x = torch.tensor([1, 2, 3])        x = tf.constant([1, 2, 3])
x = torch.zeros(3, 4)              x = tf.zeros((3, 4))
x = torch.ones(3, 4)               x = tf.ones((3, 4))
x = torch.randn(3, 4)              x = tf.random.normal((3, 4))
x = torch.arange(10)               x = tf.range(10)
```

### Model Definition

```python
# PyTorch                          
class Model(nn.Module):            
    def __init__(self):            
        super().__init__()         
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):          
        x = torch.relu(self.fc1(x))
        return self.fc2(x)         

# TensorFlow/Keras
model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])
# OR Functional API
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs, outputs)
```

### Training Loop

```python
# PyTorch
model = Model()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# TensorFlow
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=epochs)
```

### GPU Usage

```python
# PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

# TensorFlow
gpus = tf.config.list_physical_devices('GPU')
# TF automatically uses GPU if available
with tf.device('/GPU:0'):
    result = model(data)
```

### Save/Load Models

```python
# PyTorch
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# TensorFlow
model.save('model.keras')
model = keras.models.load_model('model.keras')
```

---

## 🎯 ML Algorithms Comparison

### Supervised Learning Algorithms

| Algorithm | Type | Pros | Cons | Best For |
|-----------|------|------|------|----------|
| **Linear Regression** | Regression | Simple, interpretable | Assumes linearity | Linear relationships |
| **Logistic Regression** | Classification | Fast, probabilistic | Linear boundaries | Binary classification |
| **Decision Tree** | Both | Interpretable, no scaling | Overfits easily | Understanding features |
| **Random Forest** | Both | Robust, handles overfitting | Less interpretable | General purpose |
| **Gradient Boosting** | Both | High accuracy | Slow training | Competitions, tabular |
| **SVM** | Both | Works in high dimensions | Slow for large data | Text classification |
| **KNN** | Both | Simple, no training | Slow inference | Small datasets |
| **Naive Bayes** | Classification | Fast, works with small data | Feature independence | Text classification |
| **Neural Networks** | Both | Learns complex patterns | Needs lots of data | Images, text, complex |

### Unsupervised Learning Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| **K-Means** | Clustering | Customer segmentation |
| **DBSCAN** | Clustering | Anomaly detection, spatial data |
| **Hierarchical** | Clustering | Taxonomy creation |
| **PCA** | Dim. Reduction | Feature reduction, visualization |
| **t-SNE** | Dim. Reduction | Visualization (2D/3D) |
| **UMAP** | Dim. Reduction | Visualization, faster than t-SNE |
| **Autoencoders** | Dim. Reduction | Feature learning, anomaly detection |

### Algorithm Selection Guide

```
Start Here:
    │
    ▼
Is it Supervised?
    │
    ├── YES ──► Regression or Classification?
    │               │
    │               ├── Regression ──► Linear? ──► Linear Regression
    │               │                      │
    │               │                      └── Non-linear ──► Random Forest/GBM
    │               │
    │               └── Classification ──► Binary? ──► Logistic Regression
    │                                          │
    │                                          └── Multi-class ──► Random Forest/NN
    │
    └── NO ──► Clustering or Dim. Reduction?
                    │
                    ├── Clustering ──► K-Means / DBSCAN
                    │
                    └── Dim. Reduction ──► PCA / t-SNE
```

---

## 📊 Model Evaluation Metrics

### Classification Metrics

```
Confusion Matrix:
                    Predicted
                  Pos     Neg
              ┌───────┬───────┐
Actual  Pos   │  TP   │  FN   │  → Recall = TP/(TP+FN)
              ├───────┼───────┤
        Neg   │  FP   │  TN   │  → Specificity = TN/(TN+FP)
              └───────┴───────┘
                  │       │
                  ▼       ▼
        Precision    Negative
        TP/(TP+FP)   Pred. Value

Accuracy = (TP + TN) / (TP + TN + FP + FN)
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN)/Total | Balanced classes |
| **Precision** | TP/(TP+FP) | Cost of FP high (spam) |
| **Recall** | TP/(TP+FN) | Cost of FN high (disease) |
| **F1 Score** | Harmonic mean | Imbalanced classes |
| **ROC-AUC** | Area under ROC | Overall performance |
| **PR-AUC** | Area under PR | Imbalanced, focus on positive |
| **Log Loss** | -Σ(y·log(p)) | Probabilistic predictions |

### Regression Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | Σ(y-ŷ)²/n | Penalizes large errors |
| **RMSE** | √MSE | Same units as target |
| **MAE** | Σ\|y-ŷ\|/n | Robust to outliers |
| **R² Score** | 1 - SS_res/SS_tot | Variance explained (0-1) |
| **MAPE** | Σ\|(y-ŷ)/y\|/n | Percentage error |

### Code Examples

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# Classification
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
roc_auc = roc_auc_score(y_true, y_prob)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

# Regression
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

---

## 💬 Prompt Engineering Patterns

### Basic Patterns

| Pattern | Template | Example |
|---------|----------|---------|
| **Zero-shot** | `[Task instruction]` | "Translate to French: Hello" |
| **One-shot** | `Example + Task` | "Happy→Positive. Sad→" |
| **Few-shot** | `Multiple examples + Task` | "Q:2+2=? A:4. Q:3+3=? A:" |

### Advanced Patterns

#### Chain-of-Thought (CoT)

```
Think through this step by step:

Question: A store has 45 apples. They sell 23 and receive 18 more. 
How many apples do they have?

Let's break it down:
1. Starting amount: 45 apples
2. After selling: 45 - 23 = 22 apples
3. After receiving: 22 + 18 = 40 apples

Answer: 40 apples
```

#### Role-Based Prompting

```
You are an expert Python developer with 15 years of experience.
Your task is to review the following code and suggest improvements
focusing on:
- Performance optimization
- Code readability
- Best practices

Code:
[code here]

Review:
```

#### Structured Output

```
Extract the following information from the text and format as JSON:
- Name
- Age
- Location
- Occupation

Text: "John Smith, a 35-year-old software engineer, lives in San Francisco."

Output:
{
    "name": "John Smith",
    "age": 35,
    "location": "San Francisco",
    "occupation": "software engineer"
}
```

#### Self-Consistency

```
Generate 5 different solutions to this problem.
Then analyze which answer appears most frequently.

Problem: [problem]

Solution 1: [Let model generate]
Solution 2: [Let model generate]
...
Most consistent answer: [majority vote]
```

#### ReAct (Reasoning + Acting)

```
Question: What is the population of the capital of France?

Thought 1: I need to find the capital of France first.
Action 1: Search[Capital of France]
Observation 1: Paris is the capital of France.

Thought 2: Now I need to find the population of Paris.
Action 2: Search[Population of Paris]
Observation 2: Paris has a population of about 2.1 million.

Thought 3: I have the answer.
Action 3: Finish[2.1 million]
```

### Prompt Templates by Use Case

| Use Case | Template |
|----------|----------|
| **Summarization** | "Summarize the following text in 3 bullet points:\n{text}" |
| **Translation** | "Translate the following {source_lang} text to {target_lang}:\n{text}" |
| **Code Generation** | "Write a {language} function that {description}. Include docstrings and comments." |
| **Classification** | "Classify the following text into one of these categories: {categories}\n\nText: {text}\n\nCategory:" |
| **Q&A** | "Context: {context}\n\nQuestion: {question}\n\nAnswer:" |
| **Extraction** | "Extract all {entity_type} from the following text:\n{text}\n\n{entity_type}:" |

### Tips for Better Prompts

```
✅ DO:
- Be specific and clear
- Provide examples when possible
- Specify output format
- Break complex tasks into steps
- Set role/persona when helpful

❌ DON'T:
- Use vague instructions
- Assume context
- Ask multiple unrelated things
- Forget to specify constraints
- Make prompts too long
```

---

<div align="center">

## 📚 Continue Learning

| Section | Link |
|---------|------|
| 📚 Notes | [Browse Notes →](../notes/README.md) |
| 💻 Code Examples | [Browse Code →](../code/README.md) |
| 🔗 Resources | [Browse Resources →](../resources/README.md) |
| 📖 Glossary | [Browse Glossary →](../glossary/README.md) |

---

[← Back to Main](../README.md)

</div>
