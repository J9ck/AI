# 🐍 Python for Machine Learning Cheatsheet

Quick reference for common Python patterns in ML.

## NumPy Essentials

```python
import numpy as np

# Array Creation
np.array([1, 2, 3])                    # From list
np.zeros((3, 4))                        # Zero matrix
np.ones((3, 4))                         # Ones matrix
np.eye(3)                               # Identity matrix
np.random.randn(3, 4)                   # Random normal
np.arange(0, 10, 2)                     # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)                    # [0, 0.25, 0.5, 0.75, 1]

# Array Operations
arr.shape                               # Dimensions
arr.reshape(2, 3)                       # Reshape
arr.T                                   # Transpose
arr.flatten()                           # 1D array

# Math Operations
np.dot(a, b)                            # Matrix multiplication
a @ b                                   # Matrix multiplication (Python 3.5+)
np.sum(arr, axis=0)                     # Sum along axis
np.mean(arr, axis=1)                    # Mean along axis
np.std(arr)                             # Standard deviation
np.max(arr), np.min(arr)                # Max/min
np.argmax(arr), np.argmin(arr)          # Index of max/min

# Indexing & Slicing
arr[1:3, 2:4]                           # Slice rows 1-2, cols 2-3
arr[arr > 0]                            # Boolean indexing
arr[[0, 2, 4]]                          # Fancy indexing

# Broadcasting
arr + 5                                 # Add scalar to all elements
arr * np.array([1, 2, 3])               # Broadcast along axis
```

## Pandas Essentials

```python
import pandas as pd

# DataFrame Creation
df = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': ['a', 'b', 'c']
})
df = pd.read_csv('file.csv')            # From CSV

# Basic Operations
df.head(10)                             # First 10 rows
df.tail(5)                              # Last 5 rows
df.info()                               # Data types, non-null counts
df.describe()                           # Statistics
df.shape                                # (rows, columns)
df.columns                              # Column names
df.dtypes                               # Data types

# Selection
df['col1']                              # Single column (Series)
df[['col1', 'col2']]                    # Multiple columns (DataFrame)
df.loc[0:5, 'col1':'col3']              # By label
df.iloc[0:5, 0:3]                       # By position
df[df['col1'] > 0]                      # Boolean filtering
df.query('col1 > 0 and col2 == "a"')    # Query syntax

# Data Manipulation
df['new_col'] = df['col1'] * 2          # Add column
df.drop('col1', axis=1)                 # Drop column
df.drop([0, 1], axis=0)                 # Drop rows
df.rename(columns={'old': 'new'})       # Rename
df.sort_values('col1', ascending=False) # Sort
df.groupby('col1').mean()               # Group by
df.pivot_table(values='val', index='a', columns='b')

# Missing Values
df.isnull().sum()                       # Count nulls
df.fillna(0)                            # Fill with value
df.dropna()                             # Drop rows with nulls
df.fillna(df.mean())                    # Fill with mean

# Merging
pd.concat([df1, df2])                   # Vertical concat
pd.merge(df1, df2, on='key')            # Join on column
df1.join(df2, how='left')               # Join on index
```

## Scikit-learn Patterns

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Don't fit on test!

# Model Training Pattern
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Cross-Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")

# Pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)
```

## Common Models Quick Reference

```python
# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
```

## Evaluation Metrics

```python
# Classification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print(classification_report(y_test, y_pred))

# Regression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

🌐 [Back to Cheatsheets](.) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
