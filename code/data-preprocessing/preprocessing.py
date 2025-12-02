"""
Data Preprocessing Utilities
============================

This module provides common data preprocessing functions for ML projects.

Topics covered:
- Handling missing values
- Feature scaling (StandardScaler, MinMaxScaler)
- Encoding categorical variables
- Feature engineering
- Train/test splitting
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union


class DataCleaner:
    """
    Utility class for cleaning and preprocessing data.
    """
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str = 'mean',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        strategy : str
            Strategy for handling missing values:
            - 'mean': Fill with mean (numeric columns)
            - 'median': Fill with median (numeric columns)
            - 'mode': Fill with mode (all columns)
            - 'drop': Drop rows with missing values
            - 'ffill': Forward fill
            - 'bfill': Backward fill
        columns : list, optional
            Specific columns to process. If None, all columns.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if columns is None:
            columns = df.columns.tolist()
        
        if strategy == 'drop':
            return df.dropna(subset=columns)
        
        for col in columns:
            if df[col].isna().sum() == 0:
                continue
                
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'ffill':
                df[col].fillna(method='ffill', inplace=True)
            elif strategy == 'bfill':
                df[col].fillna(method='bfill', inplace=True)
        
        return df
    
    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from numeric columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        columns : list
            Columns to check for outliers
        method : str
            Method for outlier detection:
            - 'iqr': Interquartile Range method
            - 'zscore': Z-score method
        threshold : float
            Threshold for outlier detection
            - For IQR: multiplier (default 1.5)
            - For Z-score: number of standard deviations (default 3)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with outliers removed
        """
        df = df.copy()
        
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                z_scores = np.abs((df[col] - mean) / std)
                df = df[z_scores <= threshold]
        
        return df


class FeatureScaler:
    """
    Feature scaling utilities.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize scaler.
        
        Parameters
        ----------
        method : str
            Scaling method:
            - 'standard': StandardScaler (z-score normalization)
            - 'minmax': MinMaxScaler (scale to [0, 1])
            - 'robust': RobustScaler (uses median and IQR)
        """
        self.method = method
        self.params = {}
    
    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        """Fit the scaler to the data."""
        X = np.array(X)
        
        if self.method == 'standard':
            self.params['mean'] = np.mean(X, axis=0)
            self.params['std'] = np.std(X, axis=0)
            # Avoid division by zero
            self.params['std'] = np.where(self.params['std'] == 0, 1, self.params['std'])
            
        elif self.method == 'minmax':
            self.params['min'] = np.min(X, axis=0)
            self.params['max'] = np.max(X, axis=0)
            self.params['range'] = self.params['max'] - self.params['min']
            self.params['range'] = np.where(self.params['range'] == 0, 1, self.params['range'])
            
        elif self.method == 'robust':
            self.params['median'] = np.median(X, axis=0)
            self.params['q1'] = np.percentile(X, 25, axis=0)
            self.params['q3'] = np.percentile(X, 75, axis=0)
            self.params['iqr'] = self.params['q3'] - self.params['q1']
            self.params['iqr'] = np.where(self.params['iqr'] == 0, 1, self.params['iqr'])
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        X = np.array(X)
        
        if self.method == 'standard':
            return (X - self.params['mean']) / self.params['std']
        
        elif self.method == 'minmax':
            return (X - self.params['min']) / self.params['range']
        
        elif self.method == 'robust':
            return (X - self.params['median']) / self.params['iqr']
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaling."""
        X = np.array(X)
        
        if self.method == 'standard':
            return X * self.params['std'] + self.params['mean']
        
        elif self.method == 'minmax':
            return X * self.params['range'] + self.params['min']
        
        elif self.method == 'robust':
            return X * self.params['iqr'] + self.params['median']


class CategoricalEncoder:
    """
    Encode categorical variables.
    """
    
    def __init__(self, method: str = 'onehot'):
        """
        Initialize encoder.
        
        Parameters
        ----------
        method : str
            Encoding method:
            - 'onehot': One-hot encoding
            - 'label': Label encoding
            - 'ordinal': Ordinal encoding (requires order)
        """
        self.method = method
        self.mappings = {}
    
    def fit(self, X: Union[pd.Series, np.ndarray], order: Optional[List] = None) -> 'CategoricalEncoder':
        """Fit the encoder."""
        X = pd.Series(X)
        unique_values = X.unique()
        
        if self.method == 'label':
            self.mappings = {val: i for i, val in enumerate(sorted(unique_values))}
            
        elif self.method == 'ordinal' and order is not None:
            self.mappings = {val: i for i, val in enumerate(order)}
            
        elif self.method == 'onehot':
            self.mappings = {val: i for i, val in enumerate(sorted(unique_values))}
        
        return self
    
    def transform(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Transform the data."""
        X = pd.Series(X)
        
        if self.method in ['label', 'ordinal']:
            return np.array([self.mappings.get(x, -1) for x in X])
        
        elif self.method == 'onehot':
            n_samples = len(X)
            n_categories = len(self.mappings)
            result = np.zeros((n_samples, n_categories))
            
            for i, val in enumerate(X):
                if val in self.mappings:
                    result[i, self.mappings[val]] = 1
            
            return result
    
    def fit_transform(self, X: Union[pd.Series, np.ndarray], **kwargs) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, **kwargs).transform(X)


def create_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Create polynomial features.
    
    Parameters
    ----------
    X : np.ndarray
        Input features, shape (n_samples, n_features)
    degree : int
        Polynomial degree
        
    Returns
    -------
    np.ndarray
        Polynomial features
    """
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # Start with original features
    result = [X]
    
    # Add polynomial terms
    for d in range(2, degree + 1):
        result.append(X ** d)
    
    # Add interaction terms (for degree 2)
    if degree >= 2 and n_features > 1:
        interactions = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions.append(X[:, i:i+1] * X[:, j:j+1])
        if interactions:
            result.append(np.hstack(interactions))
    
    return np.hstack(result)


def train_test_split(
    *arrays,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> List[np.ndarray]:
    """
    Split arrays into train and test sets.
    
    Parameters
    ----------
    *arrays : sequence of arrays
        Arrays to split (must have same length)
    test_size : float
        Proportion of data for test set
    random_state : int, optional
        Random seed for reproducibility
    shuffle : bool
        Whether to shuffle before splitting
        
    Returns
    -------
    list
        [train_1, test_1, train_2, test_2, ...]
    """
    n_samples = len(arrays[0])
    
    # Verify all arrays have same length
    for arr in arrays:
        if len(arr) != n_samples:
            raise ValueError("All arrays must have the same length")
    
    # Create indices
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
    
    # Split point
    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Split arrays
    result = []
    for arr in arrays:
        arr = np.array(arr)
        result.append(arr[train_indices])
        result.append(arr[test_indices])
    
    return result


def demo_preprocessing():
    """Demonstrate preprocessing utilities."""
    
    print("=" * 60)
    print("DATA PREPROCESSING DEMO")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
    })
    
    # Add some missing values
    df.loc[np.random.choice(n_samples, 10, replace=False), 'age'] = np.nan
    df.loc[np.random.choice(n_samples, 5, replace=False), 'income'] = np.nan
    
    # Add outliers
    df.loc[0, 'income'] = 500000  # Outlier
    
    print("\n1. Original Data")
    print("-" * 40)
    print(df.head())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Handle missing values
    print("\n2. Handle Missing Values (mean)")
    print("-" * 40)
    cleaner = DataCleaner()
    df_clean = cleaner.handle_missing_values(df, strategy='mean')
    print(f"Missing values after cleaning:\n{df_clean.isnull().sum()}")
    
    # Remove outliers
    print("\n3. Remove Outliers (IQR method)")
    print("-" * 40)
    print(f"Shape before: {df_clean.shape}")
    df_no_outliers = cleaner.remove_outliers(df_clean, columns=['income'], method='iqr')
    print(f"Shape after: {df_no_outliers.shape}")
    
    # Feature scaling
    print("\n4. Feature Scaling")
    print("-" * 40)
    
    numeric_data = df_no_outliers[['age', 'income']].values
    
    # StandardScaler
    standard_scaler = FeatureScaler(method='standard')
    scaled_standard = standard_scaler.fit_transform(numeric_data)
    print(f"StandardScaler - Mean: {scaled_standard.mean(axis=0).round(4)}, Std: {scaled_standard.std(axis=0).round(4)}")
    
    # MinMaxScaler
    minmax_scaler = FeatureScaler(method='minmax')
    scaled_minmax = minmax_scaler.fit_transform(numeric_data)
    print(f"MinMaxScaler - Min: {scaled_minmax.min(axis=0).round(4)}, Max: {scaled_minmax.max(axis=0).round(4)}")
    
    # Categorical encoding
    print("\n5. Categorical Encoding")
    print("-" * 40)
    
    # Label encoding
    label_encoder = CategoricalEncoder(method='label')
    education_encoded = label_encoder.fit_transform(df_no_outliers['education'])
    print(f"Label Encoding (education): {df_no_outliers['education'].values[:5]} -> {education_encoded[:5]}")
    
    # One-hot encoding
    onehot_encoder = CategoricalEncoder(method='onehot')
    city_onehot = onehot_encoder.fit_transform(df_no_outliers['city'])
    print(f"One-Hot Encoding (city) shape: {city_onehot.shape}")
    print(f"Categories: {list(onehot_encoder.mappings.keys())}")
    
    # Polynomial features
    print("\n6. Polynomial Features")
    print("-" * 40)
    X_simple = np.array([[1, 2], [3, 4], [5, 6]])
    X_poly = create_polynomial_features(X_simple, degree=2)
    print(f"Original shape: {X_simple.shape}")
    print(f"Polynomial shape: {X_poly.shape}")
    print(f"Features: original, squared, interactions")
    
    # Train-test split
    print("\n7. Train-Test Split")
    print("-" * 40)
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, 100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    return df_clean


if __name__ == "__main__":
    demo_preprocessing()
