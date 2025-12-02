# 📊 NumPy & Pandas Cheatsheet

Quick reference for data manipulation with NumPy and Pandas.

## NumPy

### Array Creation

```python
import numpy as np

# Basic creation
np.array([1, 2, 3])                      # 1D array
np.array([[1, 2], [3, 4]])               # 2D array
np.zeros((3, 4))                         # 3×4 zeros
np.ones((2, 3, 4))                       # 2×3×4 ones
np.full((2, 3), 7)                       # Fill with 7
np.eye(4)                                # 4×4 identity
np.empty((2, 3))                         # Uninitialized

# Ranges
np.arange(0, 10, 2)                      # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)                     # 5 points from 0 to 1

# Random
np.random.rand(3, 4)                     # Uniform [0, 1)
np.random.randn(3, 4)                    # Standard normal
np.random.randint(0, 10, (3, 4))         # Random integers
np.random.choice([1, 2, 3], 5)           # Random selection
np.random.seed(42)                       # Set seed
```

### Array Operations

```python
# Arithmetic
a + b, a - b, a * b, a / b               # Element-wise
a @ b                                    # Matrix multiply
np.dot(a, b)                             # Dot product
np.matmul(a, b)                          # Matrix multiply

# Math functions
np.sqrt(a), np.exp(a), np.log(a)
np.sin(a), np.cos(a), np.tan(a)
np.abs(a), np.power(a, 2)

# Statistics
np.sum(a), np.mean(a), np.std(a)
np.var(a), np.median(a)
np.min(a), np.max(a)
np.argmin(a), np.argmax(a)
np.cumsum(a), np.cumprod(a)
np.percentile(a, 75)
np.corrcoef(a, b)

# Axis operations
np.sum(a, axis=0)                        # Sum columns
np.sum(a, axis=1)                        # Sum rows
np.sum(a, axis=-1)                       # Sum last axis
```

### Reshaping & Manipulation

```python
# Reshaping
a.reshape(2, 6)                          # New shape
a.flatten()                              # 1D copy
a.ravel()                                # 1D view
a.T                                      # Transpose
a.swapaxes(0, 1)                         # Swap axes

# Adding/removing dimensions
np.expand_dims(a, axis=0)                # Add dimension
np.squeeze(a)                            # Remove size-1 dims
a[np.newaxis, :]                         # Add dimension
a[:, np.newaxis]                         # Add dimension

# Stacking
np.vstack([a, b])                        # Vertical stack
np.hstack([a, b])                        # Horizontal stack
np.concatenate([a, b], axis=0)           # Along axis
np.split(a, 3)                           # Split into 3
```

### Indexing & Slicing

```python
# Basic indexing
a[0]                                     # First element
a[-1]                                    # Last element
a[1:4]                                   # Elements 1-3
a[::2]                                   # Every 2nd element
a[::-1]                                  # Reverse

# 2D indexing
a[1, 2]                                  # Single element
a[1:3, 2:4]                              # Subarray
a[:, 1]                                  # All rows, col 1
a[1, :]                                  # Row 1, all cols

# Boolean indexing
a[a > 0]                                 # Where a > 0
a[(a > 0) & (a < 5)]                     # Multiple conditions
np.where(a > 0, a, 0)                    # Conditional replace

# Fancy indexing
a[[0, 2, 4]]                             # Select by index
a[np.array([True, False, True])]         # Boolean array
```

### Linear Algebra

```python
from numpy.linalg import inv, det, eig, svd, norm

np.linalg.inv(a)                         # Inverse
np.linalg.det(a)                         # Determinant
np.linalg.eig(a)                         # Eigenvalues/vectors
np.linalg.svd(a)                         # SVD
np.linalg.norm(a)                        # Norm
np.linalg.solve(A, b)                    # Solve Ax = b
```

---

## Pandas

### Series & DataFrame Creation

```python
import pandas as pd

# Series
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s = pd.Series({'a': 1, 'b': 2})

# DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['x', 'y', 'z']
})
df = pd.DataFrame(np_array, columns=['A', 'B'])

# From files
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')
df = pd.read_json('file.json')
df = pd.read_parquet('file.parquet')

# To files
df.to_csv('file.csv', index=False)
df.to_excel('file.xlsx', index=False)
df.to_parquet('file.parquet')
```

### Viewing Data

```python
df.head(10)                              # First 10 rows
df.tail(5)                               # Last 5 rows
df.sample(5)                             # Random 5 rows
df.info()                                # Types, non-null
df.describe()                            # Statistics
df.shape                                 # (rows, cols)
df.columns                               # Column names
df.dtypes                                # Data types
df.memory_usage()                        # Memory per column
```

### Selection

```python
# Columns
df['col']                                # Single column (Series)
df[['col1', 'col2']]                     # Multiple columns (DF)
df.col                                   # Attribute access

# Rows by label
df.loc[0]                                # Row with label 0
df.loc[0:5]                              # Rows 0-5 (inclusive!)
df.loc[0:5, 'A':'C']                     # Rows 0-5, cols A-C
df.loc[df['A'] > 0]                      # Boolean selection

# Rows by position
df.iloc[0]                               # First row
df.iloc[0:5]                             # Rows 0-4
df.iloc[0:5, 0:3]                        # Rows 0-4, cols 0-2
df.iloc[[0, 2, 4]]                       # Specific rows

# Boolean indexing
df[df['A'] > 0]                          # Where A > 0
df[(df['A'] > 0) & (df['B'] < 10)]       # Multiple conditions
df.query('A > 0 and B < 10')             # Query string
```

### Data Manipulation

```python
# Adding columns
df['new'] = df['A'] + df['B']
df['new'] = df['A'].apply(lambda x: x * 2)
df.assign(new=df['A'] * 2)               # Returns new df

# Removing
df.drop('col', axis=1)                   # Drop column
df.drop([0, 1], axis=0)                  # Drop rows
df.drop(columns=['col1', 'col2'])        # Drop columns

# Renaming
df.rename(columns={'old': 'new'})
df.columns = ['A', 'B', 'C']             # Set all names

# Sorting
df.sort_values('A')                      # Sort by column
df.sort_values(['A', 'B'], ascending=[True, False])
df.sort_index()                          # Sort by index

# Filtering
df.filter(items=['A', 'B'])              # Select columns
df.filter(regex='^col')                  # Regex match
```

### Missing Data

```python
# Detection
df.isnull()                              # Boolean mask
df.isna().sum()                          # Count per column
df.isnull().sum().sum()                  # Total nulls

# Handling
df.dropna()                              # Drop rows with any null
df.dropna(subset=['A', 'B'])             # Only check certain cols
df.fillna(0)                             # Fill with value
df.fillna(method='ffill')                # Forward fill
df.fillna(df.mean())                     # Fill with mean
df.interpolate()                         # Linear interpolation
```

### Aggregation & Grouping

```python
# Basic aggregation
df['A'].sum()
df['A'].mean()
df.agg(['sum', 'mean', 'std'])
df.agg({'A': 'sum', 'B': 'mean'})

# Group by
df.groupby('category').mean()
df.groupby(['cat1', 'cat2']).sum()
df.groupby('category').agg({
    'A': 'sum',
    'B': ['mean', 'std']
})

# Transform (returns same shape)
df['A_centered'] = df.groupby('cat')['A'].transform(
    lambda x: x - x.mean()
)

# Pivot tables
df.pivot_table(
    values='value',
    index='row_cat',
    columns='col_cat',
    aggfunc='mean'
)
```

### Merging & Joining

```python
# Concatenation
pd.concat([df1, df2])                    # Vertical (rows)
pd.concat([df1, df2], axis=1)            # Horizontal (cols)

# Merge (SQL-style join)
pd.merge(df1, df2, on='key')             # Inner join
pd.merge(df1, df2, on='key', how='left') # Left join
pd.merge(df1, df2, left_on='a', right_on='b')

# Join (on index)
df1.join(df2, how='outer')
```

### String Operations

```python
df['col'].str.lower()                    # Lowercase
df['col'].str.upper()                    # Uppercase
df['col'].str.strip()                    # Remove whitespace
df['col'].str.replace('a', 'b')          # Replace
df['col'].str.contains('pattern')        # Search
df['col'].str.split('_')                 # Split
df['col'].str.extract(r'(\d+)')          # Regex extract
df['col'].str.len()                      # Length
```

### Date/Time Operations

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
df['date'].dt.dayofweek
df['date'].dt.hour

# Time operations
df['date'] + pd.Timedelta(days=1)
df.set_index('date').resample('M').mean()  # Monthly resample
df.rolling(window=7).mean()                # Rolling average
```

### Common Patterns

```python
# Value counts
df['col'].value_counts()
df['col'].value_counts(normalize=True)

# Unique values
df['col'].unique()
df['col'].nunique()

# Apply functions
df['col'].apply(func)
df.apply(func, axis=1)                   # Row-wise
df.applymap(func)                        # Element-wise

# Change dtypes
df['col'].astype('int')
df['col'].astype('category')
pd.to_numeric(df['col'], errors='coerce')
```

---

🌐 [Back to Cheatsheets](.) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
