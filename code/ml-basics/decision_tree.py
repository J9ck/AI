"""
Decision Tree Implementation
============================

This module demonstrates decision tree classification from scratch 
and using scikit-learn.

Topics covered:
- Information Gain and Gini Impurity
- Recursive tree building
- Prediction traversal
- Visualization
"""

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


class Node:
    """
    A node in the decision tree.
    
    Attributes
    ----------
    feature : int or None
        Index of feature to split on (None for leaf nodes)
    threshold : float or None
        Threshold value for split (None for leaf nodes)
    left : Node or None
        Left child (samples where feature <= threshold)
    right : Node or None
        Right child (samples where feature > threshold)
    value : int or None
        Class label for leaf nodes (None for internal nodes)
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None


class DecisionTreeScratch:
    """
    Decision Tree Classifier implemented from scratch.
    
    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree
    min_samples_split : int
        Minimum samples required to split a node
    criterion : str
        'gini' or 'entropy' for split criterion
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_classes = None
    
    def fit(self, X, y):
        """Build the decision tree."""
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y, depth=0)
        return self
    
    def _grow_tree(self, X, y, depth):
        """Recursively grow the tree."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y, n_features)
        
        # If no valid split found, make leaf node
        if best_feature is None:
            return Node(value=self._most_common_label(y))
        
        # Split data
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # Recursively build children
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left, right=right)
    
    def _best_split(self, X, y, n_features):
        """Find the best feature and threshold to split on."""
        best_gain = -1
        best_feature, best_threshold = None, None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._information_gain(X[:, feature], y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, feature_column, y, threshold):
        """Calculate information gain from a split."""
        # Parent impurity
        parent_impurity = self._impurity(y)
        
        # Split
        left_idxs = feature_column <= threshold
        right_idxs = ~left_idxs
        
        if sum(left_idxs) == 0 or sum(right_idxs) == 0:
            return 0
        
        # Weighted child impurity
        n = len(y)
        n_left, n_right = sum(left_idxs), sum(right_idxs)
        
        child_impurity = (n_left / n * self._impurity(y[left_idxs]) +
                         n_right / n * self._impurity(y[right_idxs]))
        
        return parent_impurity - child_impurity
    
    def _impurity(self, y):
        """Calculate impurity (Gini or Entropy)."""
        hist = np.bincount(y)
        ps = hist / len(y)
        
        if self.criterion == 'gini':
            # Gini impurity: 1 - sum(p_i^2)
            return 1 - np.sum(ps ** 2)
        else:
            # Entropy: -sum(p_i * log(p_i))
            ps = ps[ps > 0]  # Avoid log(0)
            return -np.sum(ps * np.log2(ps))
    
    def _most_common_label(self, y):
        """Return the most common class label."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict class labels for samples."""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction for single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def print_tree(self, node=None, depth=0, prefix=""):
        """Print tree structure."""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            print(f"{prefix}└── Leaf: Class {node.value}")
        else:
            print(f"{prefix}├── Feature {node.feature} <= {node.threshold:.2f}")
            self.print_tree(node.left, depth + 1, prefix + "│   ")
            self.print_tree(node.right, depth + 1, prefix + "│   ")


def demo_decision_tree():
    """Demonstrate decision tree implementations."""
    
    print("=" * 60)
    print("DECISION TREE CLASSIFIER DEMO")
    print("=" * 60)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset: Iris")
    print(f"  Features: {feature_names}")
    print(f"  Classes: {list(class_names)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # ========== From Scratch Implementation ==========
    print("\n" + "-" * 40)
    print("FROM SCRATCH IMPLEMENTATION")
    print("-" * 40)
    
    model_scratch = DecisionTreeScratch(max_depth=5, min_samples_split=2)
    model_scratch.fit(X_train, y_train)
    
    y_pred_scratch = model_scratch.predict(X_test)
    accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
    
    print(f"\nAccuracy: {accuracy_scratch:.4f}")
    print(f"\nTree Structure:")
    model_scratch.print_tree()
    
    # ========== Scikit-learn Implementation ==========
    print("\n" + "-" * 40)
    print("SCIKIT-LEARN IMPLEMENTATION")
    print("-" * 40)
    
    model_sklearn = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
    model_sklearn.fit(X_train, y_train)
    
    y_pred_sklearn = model_sklearn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    print(f"\nAccuracy: {accuracy_sklearn:.4f}")
    print(f"\nFeature Importances:")
    for name, importance in zip(feature_names, model_sklearn.feature_importances_):
        print(f"  {name}: {importance:.4f}")
    
    # ========== Comparison ==========
    print("\n" + "-" * 40)
    print("COMPARISON")
    print("-" * 40)
    print(f"\n{'Model':<20} {'Accuracy':>10}")
    print("-" * 30)
    print(f"{'From Scratch':<20} {accuracy_scratch:>10.4f}")
    print(f"{'Scikit-learn':<20} {accuracy_sklearn:>10.4f}")
    
    print("\n" + "-" * 40)
    print("CLASSIFICATION REPORT (Scikit-learn)")
    print("-" * 40)
    print(classification_report(y_test, y_pred_sklearn, target_names=class_names))
    
    return model_scratch, model_sklearn


if __name__ == "__main__":
    demo_decision_tree()
