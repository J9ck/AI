# 🎯 Machine Learning Fundamentals

## Table of Contents
- [What is Machine Learning?](#what-is-machine-learning)
- [Types of Machine Learning](#types-of-machine-learning)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Model Evaluation](#model-evaluation)
- [Feature Engineering](#feature-engineering)
- [Practical Tips](#practical-tips)

---

## What is Machine Learning?

Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE ML PARADIGM SHIFT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TRADITIONAL PROGRAMMING          MACHINE LEARNING             │
│   ┌──────────────────┐            ┌──────────────────┐         │
│   │      Data        │            │      Data        │         │
│   └────────┬─────────┘            └────────┬─────────┘         │
│            │                               │                    │
│   ┌────────▼─────────┐            ┌────────▼─────────┐         │
│   │     Program      │            │     Answers      │         │
│   └────────┬─────────┘            └────────┬─────────┘         │
│            │                               │                    │
│            ▼                               ▼                    │
│   ┌──────────────────┐            ┌──────────────────┐         │
│   │     Answers      │            │     Program      │         │
│   └──────────────────┘            │    (Model)       │         │
│                                   └──────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Types of Machine Learning

### 1. Supervised Learning
- Learning from labeled data
- **Goal**: Learn a mapping from inputs to outputs
- **Examples**: Classification, Regression

### 2. Unsupervised Learning
- Learning from unlabeled data
- **Goal**: Find hidden patterns
- **Examples**: Clustering, Dimensionality Reduction

### 3. Semi-Supervised Learning
- Combination of labeled and unlabeled data
- Useful when labeling is expensive

### 4. Reinforcement Learning
- Learning through interaction with environment
- **Goal**: Maximize cumulative reward

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ML TAXONOMY                                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                           Machine Learning                              │
│                                 │                                       │
│     ┌───────────────┬──────────┼──────────┬───────────────┐           │
│     │               │          │          │               │           │
│     ▼               ▼          ▼          ▼               ▼           │
│ Supervised    Unsupervised   Semi-    Reinforcement   Self-          │
│  Learning      Learning     Supervised  Learning      Supervised      │
│     │               │                       │             │           │
│     │               │                       │             │           │
│ ┌───┴───┐     ┌────┴────┐              ┌───┴───┐     ┌───┴───┐       │
│ │       │     │         │              │       │     │       │       │
│ Class.  Reg.  Cluster.  Dim.Red.    Policy  Value  Contras. Masked   │
│                                       Based  Based   Learn.  AutoEnc.│
└────────────────────────────────────────────────────────────────────────┘
```

---

## Supervised Learning

### Classification Algorithms

#### 1. Logistic Regression
Despite its name, used for binary classification.

**Sigmoid Function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Loss Function (Binary Cross-Entropy):**
$$L = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

#### 2. Decision Trees
Tree-based model using feature splits.

```
                    [Is Spam?]
                        │
              ┌─────────┴─────────┐
              │                   │
        Contains "$$$"?     Has attachments?
              │                   │
         ┌────┴────┐         ┌────┴────┐
         │         │         │         │
       Spam    Check      Normal    Check
               sender               links
```

**Key Concepts:**
- **Gini Impurity**: $G = 1 - \sum_{i=1}^{C} p_i^2$
- **Information Gain**: $IG = H(parent) - \sum \frac{n_{child}}{n_{parent}} H(child)$
- **Entropy**: $H = -\sum_{i=1}^{C} p_i \log_2(p_i)$

#### 3. Random Forest
Ensemble of decision trees using bagging.

**Key Hyperparameters:**
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split
- `max_features`: Features considered per split

#### 4. Support Vector Machines (SVM)
Finds optimal hyperplane for classification.

**Objective (Soft Margin):**
$$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

**Kernel Trick:**
- Linear: $K(x, x') = x \cdot x'$
- Polynomial: $K(x, x') = (x \cdot x' + c)^d$
- RBF: $K(x, x') = \exp(-\gamma||x - x'||^2)$

### Regression Algorithms

#### Linear Regression
$$\hat{y} = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$$

**Loss Function (MSE):**
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

#### Gradient Descent
$$w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}$$

**Variants:**
- **Batch GD**: Uses entire dataset
- **Stochastic GD**: Uses single sample
- **Mini-batch GD**: Uses small batches

---

## Unsupervised Learning

### Clustering

#### K-Means
1. Initialize K centroids randomly
2. Assign points to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence

**Objective:**
$$J = \sum_{i=1}^{n} \min_k ||x_i - \mu_k||^2$$

#### Hierarchical Clustering
- **Agglomerative**: Bottom-up approach
- **Divisive**: Top-down approach

### Dimensionality Reduction

#### Principal Component Analysis (PCA)
1. Standardize the data
2. Compute covariance matrix
3. Calculate eigenvectors and eigenvalues
4. Select top k eigenvectors
5. Transform data

**Variance Explained:**
$$\text{Variance Ratio} = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}$$

---

## Model Evaluation

### Classification Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | $\frac{TP + TN}{Total}$ | Balanced classes |
| **Precision** | $\frac{TP}{TP + FP}$ | Cost of false positives high |
| **Recall** | $\frac{TP}{TP + FN}$ | Cost of false negatives high |
| **F1-Score** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Imbalanced classes |
| **AUC-ROC** | Area under ROC curve | Binary classification |

### Confusion Matrix
```
                    PREDICTED
                  Pos     Neg
              ┌───────┬───────┐
         Pos  │  TP   │  FN   │
ACTUAL        ├───────┼───────┤
         Neg  │  FP   │  TN   │
              └───────┴───────┘
```

### Cross-Validation

**K-Fold Cross-Validation:**
```
Fold 1: [TEST] [Train] [Train] [Train] [Train]
Fold 2: [Train] [TEST] [Train] [Train] [Train]
Fold 3: [Train] [Train] [TEST] [Train] [Train]
Fold 4: [Train] [Train] [Train] [TEST] [Train]
Fold 5: [Train] [Train] [Train] [Train] [TEST]
```

### Bias-Variance Tradeoff

```
Error = Bias² + Variance + Irreducible Error

        High Variance (Overfitting)
              ↑
              │    ╭─────────────╮
              │   ╱               ╲
    Error     │  ╱                 ╲  Total Error
              │ ╱   ╲_______________╲
              │╱  Variance   ╲
              │╲              ╲
              │ ╲    Bias²    ╲
              │  ╲______________╲
              └───────────────────────→
                    Model Complexity
                   
        High Bias (Underfitting)
```

---

## Feature Engineering

### Techniques

1. **Scaling**
   - StandardScaler: $z = \frac{x - \mu}{\sigma}$
   - MinMaxScaler: $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$

2. **Encoding**
   - One-Hot Encoding for categorical variables
   - Label Encoding for ordinal variables

3. **Feature Creation**
   - Polynomial features
   - Domain-specific features
   - Interaction terms

4. **Feature Selection**
   - Filter methods (correlation)
   - Wrapper methods (RFE)
   - Embedded methods (L1 regularization)

---

## Practical Tips

### 1. Start Simple
Always start with a simple baseline model (logistic regression, decision tree).

### 2. Understand Your Data
- Check for missing values
- Look at distributions
- Identify outliers
- Understand feature correlations

### 3. Prevent Overfitting
- Use cross-validation
- Apply regularization (L1, L2)
- Early stopping
- Ensemble methods

### 4. Hyperparameter Tuning
- Grid Search for small search spaces
- Random Search for larger spaces
- Bayesian Optimization for expensive models

### 5. Monitor for Data Leakage
- Keep test data separate
- Apply preprocessing within CV folds
- Be careful with time-series data

---

## Resources

- 📚 **Book**: "Pattern Recognition and Machine Learning" by Christopher Bishop
- 🎓 **Course**: Stanford CS229 - Machine Learning
- 📄 **Paper**: "A Few Useful Things to Know About Machine Learning" by Pedro Domingos

---

🌐 [Back to Notes](README.md) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
