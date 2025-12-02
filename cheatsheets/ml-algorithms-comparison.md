# 📊 ML Algorithms Comparison

Quick reference comparing machine learning algorithms.

## Classification Algorithms

| Algorithm | Pros | Cons | Best For | Complexity |
|-----------|------|------|----------|------------|
| **Logistic Regression** | Fast, interpretable, probabilistic | Linear boundaries only | Binary classification, baseline | O(nd) |
| **Decision Tree** | Interpretable, handles non-linear | Overfits easily | Feature importance, explainability | O(n log n × d) |
| **Random Forest** | Robust, handles non-linear, low overfitting | Slow inference, black box | General purpose, tabular data | O(k × n log n × d) |
| **Gradient Boosting** | High accuracy, handles various data types | Slow training, tuning required | Kaggle competitions, structured data | O(k × n × d) |
| **SVM** | Effective in high dimensions, memory efficient | Slow on large datasets, kernel choice | Text classification, high-dim data | O(n² to n³) |
| **KNN** | Simple, no training | Slow inference, curse of dimensionality | Small datasets, prototyping | O(nd) per query |
| **Naive Bayes** | Fast, works with small data, interpretable | Assumes feature independence | Text classification, spam detection | O(nd) |
| **Neural Network** | Universal approximator, handles complex patterns | Requires lots of data, black box | Image, text, complex patterns | Varies |

## Regression Algorithms

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Linear Regression** | Interpretable, fast | Linear only, sensitive to outliers | Baseline, simple relationships |
| **Ridge (L2)** | Handles multicollinearity | Still linear | When features are correlated |
| **Lasso (L1)** | Feature selection, sparse solutions | Can be unstable | When feature selection needed |
| **ElasticNet** | Combines L1 & L2 | More hyperparameters | Balanced regularization |
| **Decision Tree** | Non-linear, interpretable | Overfits, high variance | Explainability needed |
| **Random Forest** | Robust, handles non-linear | Less interpretable | General purpose |
| **Gradient Boosting** | High accuracy | Slow, overfitting risk | Competition, accuracy critical |
| **SVR** | Works well in high dim | Slow, kernel selection | High dimensional data |

## Clustering Algorithms

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **K-Means** | Fast, scalable | Requires k, spherical clusters | Large datasets, known k |
| **DBSCAN** | Finds arbitrary shapes, handles noise | Sensitive to parameters | Spatial data, unknown k |
| **Hierarchical** | No k needed, dendrogram | O(n³), doesn't scale | Small data, exploratory |
| **GMM** | Soft clustering, flexible shapes | Assumes Gaussian | Probabilistic membership |
| **Mean Shift** | No k needed, finds modes | Slow, bandwidth selection | Mode detection |

## Algorithm Selection Flowchart

```
                     ┌─────────────────────────────┐
                     │    What's your problem?     │
                     └─────────────┬───────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
    Classification            Regression              Clustering
          │                        │                        │
          ▼                        ▼                        ▼
    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │ How much    │         │ Linear      │         │ Know # of   │
    │ data?       │         │ relationship│         │ clusters?   │
    └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
           │                       │                       │
    ┌──────┴──────┐         ┌──────┴──────┐         ┌──────┴──────┐
    │             │         │             │         │             │
Small         Large       Yes           No        Yes           No
    │             │         │             │         │             │
    ▼             ▼         ▼             ▼         ▼             ▼
┌───────┐   ┌───────┐  ┌───────┐   ┌───────┐  ┌───────┐   ┌───────┐
│Naive  │   │Random │  │Linear │   │Random │  │K-Means│   │DBSCAN │
│Bayes, │   │Forest,│  │Reg,   │   │Forest,│  │       │   │Hier.  │
│SVM    │   │XGBoost│  │Ridge  │   │XGBoost│  │       │   │       │
└───────┘   └───────┘  └───────┘   └───────┘  └───────┘   └───────┘
```

## Hyperparameter Quick Reference

### Random Forest
```python
n_estimators = [100, 200, 500]      # Number of trees
max_depth = [10, 20, None]          # Tree depth
min_samples_split = [2, 5, 10]      # Min samples to split
min_samples_leaf = [1, 2, 4]        # Min samples in leaf
max_features = ['sqrt', 'log2']     # Features per split
```

### XGBoost
```python
n_estimators = [100, 200, 500]      # Number of trees
max_depth = [3, 5, 7]               # Tree depth (lower than RF)
learning_rate = [0.01, 0.1, 0.3]    # Step size
subsample = [0.8, 1.0]              # Sample ratio
colsample_bytree = [0.8, 1.0]       # Feature ratio per tree
```

### SVM
```python
C = [0.1, 1, 10, 100]               # Regularization
kernel = ['rbf', 'linear', 'poly']  # Kernel function
gamma = ['scale', 'auto', 0.1, 1]   # Kernel coefficient
```

### Neural Network
```python
hidden_layers = [(64,), (128, 64), (256, 128, 64)]
learning_rate = [0.001, 0.01]
dropout = [0.2, 0.5]
batch_size = [32, 64, 128]
```

## Metrics Quick Reference

### Classification

| Metric | Formula | Use When |
|--------|---------|----------|
| **Accuracy** | (TP+TN) / Total | Balanced classes |
| **Precision** | TP / (TP+FP) | False positives costly |
| **Recall** | TP / (TP+FN) | False negatives costly |
| **F1** | 2×(P×R)/(P+R) | Imbalanced classes |
| **ROC-AUC** | Area under ROC | Binary classification |
| **PR-AUC** | Area under PR | Highly imbalanced |

### Regression

| Metric | Formula | Notes |
|--------|---------|-------|
| **MSE** | Σ(y-ŷ)² / n | Penalizes large errors |
| **RMSE** | √MSE | Same unit as target |
| **MAE** | Σ\|y-ŷ\| / n | Robust to outliers |
| **R²** | 1 - SS_res/SS_tot | Explained variance |
| **MAPE** | Σ\|(y-ŷ)/y\| / n | Percentage error |

## Common Patterns

### Ensemble Methods
```
┌─────────────────────────────────────────────────────────────────┐
│                     ENSEMBLE METHODS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   BAGGING (Random Forest)         BOOSTING (XGBoost)           │
│   ┌─────┐ ┌─────┐ ┌─────┐        ┌─────┐                       │
│   │ T1  │ │ T2  │ │ T3  │        │ T1  │ → error               │
│   └──┬──┘ └──┬──┘ └──┬──┘        └──┬──┘    │                  │
│      │       │       │              │       ▼                  │
│      └───────┼───────┘         ┌─────┐                         │
│              │                 │ T2  │ → error                 │
│         [Average]              └──┬──┘    │                    │
│              │                    │       ▼                    │
│              ▼                    └─── [Weighted Sum]          │
│        Final Pred                        │                     │
│                                          ▼                     │
│   Parallel training           Sequential training              │
│   Reduces variance            Reduces bias                     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│   STACKING                                                      │
│   ┌─────┐ ┌─────┐ ┌─────┐                                      │
│   │ M1  │ │ M2  │ │ M3  │   Different algorithms              │
│   └──┬──┘ └──┬──┘ └──┬──┘                                      │
│      └───────┼───────┘                                         │
│              │                                                  │
│         [Meta Model]   Learns to combine predictions           │
│              │                                                  │
│              ▼                                                  │
│         Final Pred                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Model Selection Tips

1. **Start Simple**: Begin with logistic regression/linear regression
2. **Tree-Based**: Random Forest for robust baseline
3. **Boosting**: XGBoost/LightGBM for competitions
4. **Deep Learning**: When you have lots of data and compute
5. **Ensemble**: Combine different model types for best results

---

🌐 [Back to Cheatsheets](.) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
