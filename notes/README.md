# 📖 AI/ML Notes

Welcome to the notes section! This is where I document comprehensive theory and concepts in AI/ML.

## 📑 Contents

| Topic | Description | File |
|-------|-------------|------|
| 🎯 **Machine Learning Fundamentals** | Core ML concepts, algorithms, and evaluation | [machine-learning-fundamentals.md](machine-learning-fundamentals.md) |
| 🧠 **Deep Learning** | Neural networks, backpropagation, architectures | [deep-learning.md](deep-learning.md) |
| 🔄 **Transformers** | Attention mechanism, BERT, GPT, and beyond | [transformers.md](transformers.md) |
| 🗣️ **NLP** | Text processing, embeddings, language models | [nlp.md](nlp.md) |
| 👁️ **Computer Vision** | CNNs, object detection, segmentation | [computer-vision.md](computer-vision.md) |
| 🎨 **Generative AI** | GANs, VAEs, Diffusion models, LLMs | [generative-ai.md](generative-ai.md) |
| 🎮 **Reinforcement Learning** | RL fundamentals, Q-learning, policy gradient | [reinforcement-learning.md](reinforcement-learning.md) |
| ⚙️ **MLOps** | Deployment, monitoring, pipelines | [mlops.md](mlops.md) |

## 🎓 Learning Path

```
                    ┌─────────────────────────────────────┐
                    │     START YOUR AI JOURNEY HERE!     │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │    Machine Learning Fundamentals    │
                    │    (Start here if you're new!)      │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌───────────────────────┐         ┌───────────────────────┐
        │     Deep Learning     │         │ Reinforcement Learning│
        └───────────────────────┘         └───────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│      NLP      │       │Computer Vision│
└───────────────┘       └───────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │     Transformers      │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │    Generative AI      │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │        MLOps          │
        │   (When deploying)    │
        └───────────────────────┘
```

## 📚 How to Use These Notes

1. **Sequential Learning**: Follow the learning path above for a structured approach
2. **Reference**: Jump directly to topics you need to brush up on
3. **Deep Dive**: Each note contains links to papers and resources for further reading

---

🌐 [Back to Main Repository](../README.md) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
# 📚 AI/ML Notes - Concepts & Theory

> A comprehensive guide to understanding Artificial Intelligence and Machine Learning fundamentals.

[← Back to Main](../README.md)

---

## 📋 Table of Contents

- [Machine Learning Fundamentals](#-machine-learning-fundamentals)
- [Neural Networks](#-neural-networks)
- [Deep Learning Architectures](#-deep-learning-architectures)
- [Natural Language Processing](#-natural-language-processing)
- [Computer Vision](#-computer-vision)
- [Generative AI](#-generative-ai)
- [Reinforcement Learning](#-reinforcement-learning)
- [AI Ethics & Safety](#️-ai-ethics--safety)

---

## 🧠 Machine Learning Fundamentals

### Supervised vs Unsupervised Learning

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|---------------------|----------------------|
| **Data** | Labeled data (input-output pairs) | Unlabeled data |
| **Goal** | Learn mapping from inputs to outputs | Discover hidden patterns/structure |
| **Examples** | Classification, Regression | Clustering, Dimensionality Reduction |
| **Algorithms** | Linear Regression, SVM, Random Forest | K-Means, PCA, DBSCAN |

### Types of Machine Learning Problems

```
┌─────────────────────────────────────────────────────────────────┐
│                    Machine Learning Tasks                        │
├────────────────────┬────────────────────┬───────────────────────┤
│    Supervised      │   Unsupervised     │    Reinforcement      │
├────────────────────┼────────────────────┼───────────────────────┤
│ • Classification   │ • Clustering       │ • Policy Learning     │
│ • Regression       │ • Dim. Reduction   │ • Value Learning      │
│ • Ranking          │ • Anomaly Detection│ • Model-Based         │
│ • Forecasting      │ • Association      │ • Model-Free          │
└────────────────────┴────────────────────┴───────────────────────┘
```

### Regression

**Goal**: Predict continuous numerical values.

**Common Algorithms**:
- **Linear Regression**: Fits a linear relationship between features and target
  - Equation: `y = mx + b` (simple) or `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ` (multiple)
- **Polynomial Regression**: Captures non-linear relationships
- **Ridge Regression (L2)**: Linear regression with L2 regularization to prevent overfitting
- **Lasso Regression (L1)**: Linear regression with L1 regularization (can zero out features)
- **Elastic Net**: Combination of L1 and L2 regularization

**Key Metrics**:
- Mean Squared Error (MSE): `Σ(y_pred - y_actual)² / n`
- Root Mean Squared Error (RMSE): `√MSE`
- Mean Absolute Error (MAE): `Σ|y_pred - y_actual| / n`
- R² Score (Coefficient of Determination): Measures variance explained

### Classification

**Goal**: Predict discrete class labels.

**Binary Classification** (2 classes):
- Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees

**Multi-class Classification** (>2 classes):
- One-vs-Rest (OvR)
- One-vs-One (OvO)
- Softmax Regression

**Key Metrics**:
```
                    Predicted
                  Pos    Neg
              ┌───────┬───────┐
Actual  Pos   │  TP   │  FN   │
              ├───────┼───────┤
        Neg   │  FP   │  TN   │
              └───────┴───────┘

Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)  -- "Of predicted positives, how many correct?"
Recall    = TP / (TP + FN)  -- "Of actual positives, how many found?"
F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

### Clustering

**Goal**: Group similar data points without labels.

**Algorithms**:
- **K-Means**: Partitions data into K clusters based on centroid distance
  - Choose K (number of clusters)
  - Initialize centroids randomly
  - Assign points to nearest centroid
  - Update centroids to cluster mean
  - Repeat until convergence
  
- **Hierarchical Clustering**: Creates tree-like structure of clusters
  - Agglomerative (bottom-up)
  - Divisive (top-down)
  
- **DBSCAN**: Density-based clustering, finds arbitrary shaped clusters
  - Core points: Points with many neighbors
  - Border points: Near core points
  - Noise points: Neither core nor border

### Feature Engineering

**The process of using domain knowledge to create features that make ML algorithms work better.**

**Techniques**:
| Technique | Description | Example |
|-----------|-------------|---------|
| **Normalization** | Scale features to [0, 1] | `(x - min) / (max - min)` |
| **Standardization** | Scale to mean=0, std=1 | `(x - μ) / σ` |
| **One-Hot Encoding** | Convert categorical to binary | `color: [1,0,0], [0,1,0], [0,0,1]` |
| **Binning** | Convert continuous to categorical | Age groups: 0-18, 19-35, 36-50 |
| **Log Transform** | Handle skewed distributions | `log(x + 1)` |
| **Polynomial Features** | Create interaction terms | `x₁, x₂, x₁², x₂², x₁x₂` |

### Train/Test Split & Cross-Validation

```
Dataset Split Strategy:
┌─────────────────────────────────────────────────────────┐
│                    Full Dataset                         │
├─────────────────────────────┬───────────┬───────────────┤
│        Training Set         │    Val    │   Test Set    │
│           (60-70%)          │  (10-15%) │   (20-25%)    │
└─────────────────────────────┴───────────┴───────────────┘

K-Fold Cross-Validation (K=5):
┌─────┬─────┬─────┬─────┬─────┐
│ Val │Train│Train│Train│Train│  Fold 1
├─────┼─────┼─────┼─────┼─────┤
│Train│ Val │Train│Train│Train│  Fold 2
├─────┼─────┼─────┼─────┼─────┤
│Train│Train│ Val │Train│Train│  Fold 3
├─────┼─────┼─────┼─────┼─────┤
│Train│Train│Train│ Val │Train│  Fold 4
├─────┼─────┼─────┼─────┼─────┤
│Train│Train│Train│Train│ Val │  Fold 5
└─────┴─────┴─────┴─────┴─────┘
Final score = Average of all fold scores
```

### Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Error

High Bias (Underfitting):
- Model too simple
- Can't capture patterns
- High training error
- High test error

High Variance (Overfitting):
- Model too complex
- Memorizes training data
- Low training error
- High test error

┌─────────────────────────────────────────────────────────┐
│         Model Complexity vs Error                       │
│                                                         │
│  Error                                                  │
│    │                                                    │
│    │   ╲                                   ╱           │
│    │    ╲   Total Error                   ╱            │
│    │     ╲                               ╱             │
│    │      ╲─────╲                 ╱─────╱              │
│    │       Bias  ╲───────────────╱  Variance          │
│    │              ╲             ╱                      │
│    │               ╲___________╱                       │
│    │                    │                              │
│    └────────────────────┼──────────────────────────    │
│                   Optimal Complexity                   │
└─────────────────────────────────────────────────────────┘
```

**Solutions**:
- High Bias: More features, more complex model, less regularization
- High Variance: More data, simpler model, more regularization, dropout

---

## 🔮 Neural Networks

### Perceptron - The Building Block

```
        x₁ ──→ w₁ ──┐
                    │
        x₂ ──→ w₂ ──┼──→ Σ ──→ f(z) ──→ output
                    │
        x₃ ──→ w₃ ──┘
                    ↑
                   bias (b)

z = w₁x₁ + w₂x₂ + w₃x₃ + b
output = f(z)  where f is activation function
```

### Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **Sigmoid** | `σ(x) = 1 / (1 + e⁻ˣ)` | (0, 1) | Binary classification output |
| **Tanh** | `tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)` | (-1, 1) | Hidden layers (centered output) |
| **ReLU** | `max(0, x)` | [0, ∞) | Hidden layers (default choice) |
| **Leaky ReLU** | `max(0.01x, x)` | (-∞, ∞) | Prevents dying ReLU |
| **Softmax** | `eˣⁱ / Σeˣʲ` | (0, 1) | Multi-class classification output |
| **GELU** | `x · Φ(x)` | (-∞, ∞) | Transformers |

```
ReLU Graph:           Sigmoid Graph:        Tanh Graph:
    │    ╱                │    ____             │    ____
    │   ╱                 │   ╱                 │   ╱
────┼──╱────          ────┼──╱────          ────┼─╱────
    │╱                    │╱                   ╱│
    │                     │                   ╱ │
                                         ────   │
```

### Backpropagation

**The algorithm used to train neural networks by computing gradients.**

**Steps**:
1. **Forward Pass**: Compute predictions layer by layer
2. **Compute Loss**: Compare predictions with actual values
3. **Backward Pass**: Compute gradients using chain rule
4. **Update Weights**: Adjust weights using gradients

```
Chain Rule Application:
∂Loss/∂w₁ = ∂Loss/∂output × ∂output/∂z × ∂z/∂w₁
```

### Gradient Descent Variants

| Variant | Description | Batch Size |
|---------|-------------|------------|
| **Batch GD** | Use all samples per update | Full dataset |
| **Stochastic GD** | Use one sample per update | 1 |
| **Mini-batch GD** | Use subset per update | 16-256 typically |

**Update Rule**:
```
w = w - η × ∇L(w)

where:
η = learning rate
∇L(w) = gradient of loss with respect to weights
```

### Learning Rate

```
Learning Rate Effects:
                                              
Too Small:                Too Large:           Just Right:
│                         │                    │
│  ╭─────────────         │    ╱╲     ╱╲      │     ╲
│ ╱                       │   ╱  ╲   ╱  ╲     │      ╲___
│╱                        │  ╱    ╲_╱    ╲    │         ╲__
│───────────────          │ ╱             ╲   │            ╲_
(Very slow convergence)   (Divergence)        (Good convergence)
```

**Learning Rate Schedules**:
- **Step Decay**: Reduce LR by factor every N epochs
- **Exponential Decay**: `lr = lr₀ × e^(-kt)`
- **Cosine Annealing**: Smoothly decrease and optionally restart
- **Warmup**: Start small, gradually increase, then decay

### Optimizers

| Optimizer | Key Idea | Update Rule |
|-----------|----------|-------------|
| **SGD** | Basic gradient descent | `w -= lr × g` |
| **Momentum** | Accumulate velocity | `v = βv + g; w -= lr × v` |
| **RMSprop** | Adaptive learning rates | Scales by running avg of squared gradients |
| **Adam** | Momentum + RMSprop | Combines both approaches |
| **AdamW** | Adam + Weight Decay | Decoupled weight decay regularization |

**Adam** (Most commonly used):
```
m = β₁ × m + (1 - β₁) × g          # First moment (momentum)
v = β₂ × v + (1 - β₂) × g²         # Second moment (RMSprop)
m̂ = m / (1 - β₁ᵗ)                  # Bias correction
v̂ = v / (1 - β₂ᵗ)
w = w - lr × m̂ / (√v̂ + ε)

Common values: β₁=0.9, β₂=0.999, ε=1e-8
```

---

## 🏗️ Deep Learning Architectures

### Convolutional Neural Networks (CNNs)

**Key Operations**:

```
Convolution Operation:

Input:                    Kernel (3×3):           Output:
┌───┬───┬───┬───┬───┐    ┌───┬───┬───┐    
│ 1 │ 2 │ 3 │ 0 │ 1 │    │ 1 │ 0 │ 1 │          ┌───┬───┬───┐
├───┼───┼───┼───┼───┤    ├───┼───┼───┤          │12 │...│...│
│ 0 │ 1 │ 2 │ 3 │ 0 │ * │ 0 │ 1 │ 0 │    =     ├───┼───┼───┤
├───┼───┼───┼───┼───┤    ├───┼───┼───┤          │...│...│...│
│ 1 │ 0 │ 1 │ 0 │ 2 │    │ 1 │ 0 │ 1 │          └───┴───┴───┘
├───┼───┼───┼───┼───┤    └───┴───┴───┘
│ 2 │ 1 │ 0 │ 1 │ 1 │
├───┼───┼───┼───┼───┤
│ 0 │ 0 │ 1 │ 2 │ 0 │
└───┴───┴───┴───┴───┘

Pooling (Max Pool 2×2):

Input:              Output:
┌───┬───┬───┬───┐   ┌───┬───┐
│ 1 │ 3 │ 2 │ 1 │   │ 4 │ 2 │
├───┼───┼───┼───┤   ├───┼───┤
│ 4 │ 2 │ 1 │ 0 │ → │ 5 │ 3 │
├───┼───┼───┼───┤   └───┴───┘
│ 1 │ 5 │ 3 │ 2 │
├───┼───┼───┼───┤
│ 2 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┘
```

**CNN Terms**:
- **Stride**: Step size of the kernel movement
- **Padding**: Adding zeros around input to control output size
  - `VALID`: No padding (output shrinks)
  - `SAME`: Pad to keep output same size as input
- **Receptive Field**: Input region that affects a particular output

**Output Size Formula**:
```
Output = (Input - Kernel + 2×Padding) / Stride + 1
```

### Recurrent Neural Networks (RNNs)

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ↓                                         │
Input: x₁ ──→ [RNN Cell] ──→ h₁ ──→ y₁                       │
                    │                                         │
                    ↓                                         │
Input: x₂ ──→ [RNN Cell] ──→ h₂ ──→ y₂                       │
                    │                                         │
                    ↓                                         │
Input: x₃ ──→ [RNN Cell] ──→ h₃ ──→ y₃                       │
                                                              │
                                      Hidden state loops ─────┘

h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b)
```

**Problem**: Vanishing/Exploding Gradients over long sequences

### LSTMs (Long Short-Term Memory)

**Solves RNN's vanishing gradient problem using gates.**

```
┌─────────────────────────────────────────────────────────────┐
│                        LSTM Cell                             │
│                                                              │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│   │ Forget   │  │  Input   │  │  Output  │                  │
│   │  Gate    │  │  Gate    │  │  Gate    │                  │
│   │  (f_t)   │  │  (i_t)   │  │  (o_t)   │                  │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│        │             │             │                         │
│        ↓             ↓             ↓                         │
│   c_{t-1} ──→ [×] ──[+]────────────────→ c_t (cell state)   │
│                ↑     ↑                                       │
│                f_t   i_t × c̃_t                               │
│                                                              │
│   h_t = o_t × tanh(c_t)                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Gates:
- Forget Gate: What to remove from cell state
- Input Gate: What new info to add
- Output Gate: What to output from cell state
```

### GRUs (Gated Recurrent Units)

**Simplified version of LSTM with fewer gates.**

```
GRU has 2 gates instead of 3:
- Reset Gate (r_t): Controls how much past info to forget
- Update Gate (z_t): Controls how much past info to keep

Fewer parameters = faster training
Often performs comparably to LSTM
```

### Transformers

**The architecture powering modern AI (BERT, GPT, etc.)**

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer Architecture                  │
│                                                              │
│    Input                                    Output           │
│      ↓                                        ↑              │
│  [Embedding]                           [Linear + Softmax]    │
│      ↓                                        ↑              │
│  [Positional                           [Feed Forward]        │
│   Encoding]                                   ↑              │
│      ↓                                 [Add & Norm]          │
│  ┌─────────┐                                  ↑              │
│  │ Encoder │ ←─────────────────────── [Multi-Head           │
│  │  ×N     │                           Cross-Attention]     │
│  └────┬────┘                                  ↑              │
│       │                                [Add & Norm]          │
│       │                                       ↑              │
│       │     ┌───────────────────── [Masked Multi-Head       │
│       │     │                       Self-Attention]          │
│       │     ↓                                ↑              │
│       └────→ Decoder ×N ─────────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Attention Mechanism

**"Attention is All You Need" - The core innovation**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Q = Query (what we're looking for)
K = Key (what we have to match against)
V = Value (what we return)
d_k = dimension of keys (for scaling)

Multi-Head Attention:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   ┌────────┐  ┌────────┐  ┌────────┐        ┌────────┐     │
│   │ Head 1 │  │ Head 2 │  │ Head 3 │  ...   │ Head h │     │
│   └───┬────┘  └───┬────┘  └───┬────┘        └───┬────┘     │
│       │           │           │                  │          │
│       └───────────┴───────────┴──────────────────┘          │
│                          │                                   │
│                    [Concatenate]                            │
│                          │                                   │
│                    [Linear Layer]                           │
│                          │                                   │
│                       Output                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Multiple heads allow model to attend to different aspects simultaneously
```

### Self-Attention

**Each position attends to all positions in the same sequence.**

```
Input: "The cat sat on the mat"

       The   cat   sat   on   the   mat
The    0.1   0.3   0.2   0.1  0.1   0.2   ← attention weights
cat    0.1   0.4   0.2   0.1  0.1   0.1      for each word
sat    0.1   0.2   0.3   0.2  0.1   0.1
...

Each word "attends" to every other word
Learns which words are relevant to each other
```

---

## 💬 Natural Language Processing

### Tokenization

**Breaking text into smaller units (tokens).**

| Type | Example | Output |
|------|---------|--------|
| **Word** | "Hello world" | ["Hello", "world"] |
| **Character** | "Hello" | ["H", "e", "l", "l", "o"] |
| **Subword (BPE)** | "unhappiness" | ["un", "happiness"] |
| **WordPiece** | "playing" | ["play", "##ing"] |

```
BPE (Byte Pair Encoding):
1. Start with character-level vocabulary
2. Count all adjacent pairs
3. Merge most frequent pair
4. Repeat until vocabulary size reached

Handles:
- Out-of-vocabulary words
- Morphologically rich languages
- Reduces vocabulary size
```

### Word Embeddings

**Representing words as dense vectors.**

**Word2Vec**:
- **CBOW**: Predict center word from context
- **Skip-gram**: Predict context from center word

```
Example embeddings (simplified 3D):
king    = [0.8, 0.2, 0.9]
queen   = [0.7, 0.8, 0.9]
man     = [0.9, 0.1, 0.5]
woman   = [0.8, 0.7, 0.5]

king - man + woman ≈ queen  (famous analogy)
```

**GloVe (Global Vectors)**:
- Uses word co-occurrence statistics
- Combines benefits of count-based and predictive methods

### BERT (Bidirectional Encoder Representations from Transformers)

```
Architecture:
- Encoder-only Transformer
- Bidirectional context (sees left AND right)
- Pre-trained on MLM and NSP tasks

Pre-training Tasks:
┌─────────────────────────────────────────────────────────────┐
│ Masked Language Modeling (MLM):                             │
│ Input:  "The [MASK] sat on the mat"                        │
│ Output: "cat" (predict masked word)                         │
│                                                              │
│ Next Sentence Prediction (NSP):                             │
│ Input:  "[CLS] Sentence A [SEP] Sentence B [SEP]"          │
│ Output: Is B the actual next sentence? (Yes/No)            │
└─────────────────────────────────────────────────────────────┘

Models: BERT-base (110M params), BERT-large (340M params)
Variants: RoBERTa, ALBERT, DistilBERT, DeBERTa
```

### GPT Architecture (Generative Pre-trained Transformer)

```
Architecture:
- Decoder-only Transformer
- Autoregressive (left-to-right generation)
- Causal masking (can't see future tokens)

Generation Process:
Input:  "Once upon a"
Step 1: Predict "time" → "Once upon a time"
Step 2: Predict "there" → "Once upon a time there"
Step 3: Predict "was" → "Once upon a time there was"
...

Evolution:
GPT-1:   117M parameters
GPT-2:   1.5B parameters
GPT-3:   175B parameters
GPT-4:   Estimated ~1.7T parameters (MoE)
```

### Sequence-to-Sequence Models

```
Encoder-Decoder Architecture:

Input: "Hello, how are you?"
                ↓
┌─────────────────────────┐
│        Encoder          │
│  (Processes input seq)  │
└───────────┬─────────────┘
            │
     Context Vector
            │
            ↓
┌─────────────────────────┐
│        Decoder          │
│ (Generates output seq)  │
└───────────┬─────────────┘
            ↓
Output: "Bonjour, comment allez-vous?"

Applications:
- Machine Translation
- Text Summarization
- Question Answering
```

### Named Entity Recognition (NER)

**Identifying and classifying named entities in text.**

```
Input: "Apple Inc. was founded by Steve Jobs in California."

Output with BIO tagging:
Apple     B-ORG
Inc.      I-ORG
was       O
founded   O
by        O
Steve     B-PER
Jobs      I-PER
in        O
California B-LOC
.         O

Entity Types: PER (Person), ORG (Organization), LOC (Location), 
              DATE, TIME, MONEY, etc.
```

### Sentiment Analysis

```
Approaches:
1. Rule-based: Lexicon matching
2. ML-based: Train classifiers on labeled data
3. Deep Learning: Fine-tune BERT/RoBERTa

Example Classifications:
"I love this product!" → Positive (0.95)
"Terrible experience." → Negative (0.87)
"It's okay, I guess." → Neutral (0.62)

Advanced: Aspect-based sentiment
"The food was great but the service was terrible."
→ Food: Positive, Service: Negative
```

---

## 👁️ Computer Vision

### Image Classification

```
Input Image (224×224×3)
        ↓
┌───────────────────┐
│   Conv Layers     │  ← Feature extraction
│   (Hierarchical)  │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  Pooling Layers   │  ← Reduce dimensions
└─────────┬─────────┘
          ↓
┌───────────────────┐
│   Flatten         │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│  Fully Connected  │  ← Classification
└─────────┬─────────┘
          ↓
    Output: "Cat" (softmax over classes)
```

### Object Detection

**YOLO (You Only Look Once)**:
```
┌─────────────────────────────────────────────────────────────┐
│ YOLO Approach:                                               │
│                                                              │
│ 1. Divide image into S×S grid                               │
│ 2. Each cell predicts B bounding boxes                      │
│ 3. Each box: (x, y, w, h, confidence)                       │
│ 4. Each cell predicts class probabilities                   │
│                                                              │
│ Output: All predictions in single forward pass (FAST!)      │
│                                                              │
│ Versions: YOLOv1 → v2 → v3 → v4 → v5 → v8                  │
└─────────────────────────────────────────────────────────────┘
```

**R-CNN Family**:
```
R-CNN:
Image → Region Proposals → CNN → SVM Classification
(Slow: ~47s per image)

Fast R-CNN:
Image → CNN → Region Proposals → Classification
(Faster: ~2s per image)

Faster R-CNN:
Image → CNN → Region Proposal Network → Classification
(Even faster: ~0.2s per image)
```

### Image Segmentation

```
Types:

Semantic Segmentation:
┌─────────────────┐        ┌─────────────────┐
│   Original      │   →    │ ███ Person      │
│   Image         │        │ ░░░ Background  │
│                 │        │ ▓▓▓ Car         │
└─────────────────┘        └─────────────────┘
(Each pixel gets a class label)

Instance Segmentation:
┌─────────────────┐        ┌─────────────────┐
│   Original      │   →    │ ███ Person 1    │
│   Image         │        │ ▒▒▒ Person 2    │
│                 │        │ ▓▓▓ Car 1       │
└─────────────────┘        └─────────────────┘
(Each pixel gets class AND instance ID)

Architectures: U-Net, DeepLab, Mask R-CNN
```

### Data Augmentation

**Artificially expanding training data through transformations.**

| Augmentation | Description |
|--------------|-------------|
| **Horizontal Flip** | Mirror image left-right |
| **Rotation** | Rotate by random angle |
| **Scaling** | Zoom in/out |
| **Translation** | Shift image position |
| **Color Jitter** | Adjust brightness, contrast, saturation |
| **Random Crop** | Extract random regions |
| **Cutout** | Randomly mask square regions |
| **MixUp** | Blend two images and labels |
| **CutMix** | Replace image regions from another image |

### Transfer Learning

```
Strategy:
1. Take pre-trained model (e.g., ResNet trained on ImageNet)
2. Remove final classification layer
3. Add new layer for your task
4. Option A: Freeze base, train only new layers
   Option B: Fine-tune entire model with small learning rate

┌─────────────────────────────────────────────────────────────┐
│ Pre-trained Model          │ Your Model                     │
│                            │                                │
│ [Conv Layers - FROZEN]     │ [Conv Layers - FROZEN]        │
│         ↓                  │         ↓                      │
│ [FC Layer]                 │ [New FC Layer] ← Train this   │
│         ↓                  │         ↓                      │
│ [1000 ImageNet classes]    │ [Your 10 classes]             │
└─────────────────────────────────────────────────────────────┘

Benefits:
- Less data needed
- Faster training
- Often better performance
```

---

## ✨ Generative AI

### GANs (Generative Adversarial Networks)

```
┌─────────────────────────────────────────────────────────────┐
│                     GAN Architecture                         │
│                                                              │
│   Random        ┌─────────────┐      Generated              │
│   Noise  ──────→│  Generator  │──────→  Image               │
│   (z)           └─────────────┘          │                  │
│                                          ↓                  │
│                 Real          ┌─────────────────┐           │
│                 Images ──────→│  Discriminator  │──→ Real?  │
│                               └─────────────────┘   Fake?   │
│                                                              │
│   Training:                                                  │
│   - Generator tries to fool Discriminator                   │
│   - Discriminator tries to distinguish real from fake       │
│   - Adversarial training until equilibrium                  │
└─────────────────────────────────────────────────────────────┘

Variants:
- DCGAN: Deep Convolutional GAN
- StyleGAN: Style-based generator
- CycleGAN: Image-to-image translation
- Pix2Pix: Paired image translation
```

### VAEs (Variational Autoencoders)

```
┌─────────────────────────────────────────────────────────────┐
│                    VAE Architecture                          │
│                                                              │
│   Input    ┌─────────┐    μ, σ     ┌─────────┐   Output    │
│   Image ──→│ Encoder │──→ z ←────→│ Decoder │──→ Image    │
│            └─────────┘  (latent)   └─────────┘              │
│                                                              │
│   Key Idea: Learn latent distribution, not just encoding    │
│                                                              │
│   Loss = Reconstruction Loss + KL Divergence                │
│                                                              │
│   Applications:                                             │
│   - Image generation                                        │
│   - Anomaly detection                                       │
│   - Data compression                                        │
└─────────────────────────────────────────────────────────────┘
```

### Diffusion Models

```
Forward Process (Add Noise):
x₀ ──→ x₁ ──→ x₂ ──→ ... ──→ xₜ ──→ Pure Noise

Reverse Process (Remove Noise):
Noise ──→ xₜ₋₁ ──→ ... ──→ x₁ ──→ x₀ (Generated Image)

┌─────────────────────────────────────────────────────────────┐
│ Training:                                                    │
│ 1. Take clean image                                         │
│ 2. Add noise at random timestep t                           │
│ 3. Train model to predict the noise                         │
│                                                              │
│ Generation:                                                  │
│ 1. Start with random noise                                  │
│ 2. Iteratively denoise using trained model                  │
│ 3. Obtain final image                                       │
│                                                              │
│ Models: DDPM, Stable Diffusion, DALL-E 2, Midjourney       │
└─────────────────────────────────────────────────────────────┘
```

### LLMs (Large Language Models)

```
Key Concepts:
┌─────────────────────────────────────────────────────────────┐
│ Pre-training:                                                │
│ - Train on massive text corpus (internet, books, etc.)      │
│ - Learn general language understanding                       │
│ - Self-supervised (next token prediction)                   │
│                                                              │
│ Scaling Laws:                                                │
│ - More parameters = better performance                      │
│ - More data = better performance                            │
│ - More compute = better performance                         │
│                                                              │
│ Emergent Abilities:                                         │
│ - In-context learning                                       │
│ - Chain-of-thought reasoning                                │
│ - Few-shot learning                                         │
└─────────────────────────────────────────────────────────────┘

Notable LLMs:
- GPT-3/4 (OpenAI)
- Claude (Anthropic)
- PaLM/Gemini (Google)
- LLaMA (Meta)
- Mistral/Mixtral (Mistral AI)
```

### Fine-tuning

```
Types of Fine-tuning:

Full Fine-tuning:
- Update all model parameters
- Requires significant compute
- Risk of catastrophic forgetting

Parameter-Efficient Fine-tuning (PEFT):
┌─────────────────────────────────────────────────────────────┐
│ LoRA (Low-Rank Adaptation):                                 │
│ - Freeze original weights                                   │
│ - Add trainable low-rank matrices                          │
│ - W' = W + BA where B, A are small matrices               │
│                                                              │
│ Adapter Layers:                                             │
│ - Insert small trainable modules between frozen layers     │
│                                                              │
│ Prefix Tuning:                                              │
│ - Add trainable tokens to input                            │
│                                                              │
│ Prompt Tuning:                                              │
│ - Learn soft prompts in embedding space                    │
└─────────────────────────────────────────────────────────────┘
```

### RLHF (Reinforcement Learning from Human Feedback)

```
RLHF Pipeline:

Step 1: Supervised Fine-tuning (SFT)
- Fine-tune base model on high-quality demonstrations

Step 2: Reward Model Training
- Humans rank multiple model outputs
- Train reward model to predict human preferences

Step 3: RL Fine-tuning (PPO)
- Use reward model as reward signal
- Optimize policy to maximize reward
- KL penalty to prevent divergence from SFT model

┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Prompt ──→ [LLM] ──→ Response ──→ [Reward Model] ──→ Score │
│              ↑                                       │      │
│              └───────────────────────────────────────┘      │
│                    Update via PPO                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Prompt Engineering Techniques

| Technique | Description | Example |
|-----------|-------------|---------|
| **Zero-shot** | Direct task without examples | "Translate to French: Hello" |
| **Few-shot** | Provide examples in prompt | "Q: 2+2=? A: 4, Q: 3+3=? A: 6, Q: 5+5=?" |
| **Chain-of-Thought** | Request step-by-step reasoning | "Think step by step..." |
| **Self-Consistency** | Sample multiple responses, take majority | Generate 5 answers, vote |
| **Tree-of-Thought** | Explore multiple reasoning paths | Branch and evaluate |
| **ReAct** | Reasoning + Acting in loops | Thought → Action → Observation |

---

## 🎮 Reinforcement Learning

### Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                   RL Framework                               │
│                                                              │
│                    ┌─────────┐                              │
│                    │  Agent  │                              │
│                    └────┬────┘                              │
│                         │                                    │
│            Action (aₜ)  │   State (sₜ)                      │
│                    ↓    │    ↑                              │
│                ┌────────┴────────┐                          │
│                │   Environment   │                          │
│                └────────┬────────┘                          │
│                         │                                    │
│                    Reward (rₜ)                              │
│                                                              │
│  Goal: Learn policy π(a|s) that maximizes cumulative reward │
└─────────────────────────────────────────────────────────────┘

Key Terms:
- State (s): Current situation
- Action (a): What agent can do
- Reward (r): Feedback signal
- Policy (π): Action selection strategy
- Value (V): Expected future reward from state
- Q-value (Q): Expected future reward from state-action pair
```

### Q-Learning

```
Q-Learning Update:
Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') - Q(s,a)]

where:
α = learning rate
γ = discount factor (importance of future rewards)
r = immediate reward
s' = next state

Q-Table Example:
         Action 1   Action 2   Action 3
State 1    0.5       0.2        0.8
State 2    0.1       0.9        0.3
State 3    0.7       0.4        0.6

DQN (Deep Q-Network):
- Replace Q-table with neural network
- Experience replay buffer
- Target network for stable learning
```

### Policy Gradients

```
Direct Policy Optimization:

Instead of learning value function, directly learn policy

Policy Gradient Theorem:
∇J(θ) = 𝔼[∇log π(a|s;θ) × R]

REINFORCE Algorithm:
1. Sample trajectory using current policy
2. Compute returns for each step
3. Update policy: θ ← θ + α × ∇log π(a|s;θ) × G

Problem: High variance
Solution: Subtract baseline (e.g., value function)
```

### Actor-Critic Methods

```
┌─────────────────────────────────────────────────────────────┐
│                Actor-Critic Architecture                     │
│                                                              │
│     State ──→ ┌────────────┐                                │
│               │   Actor    │ ──→ Action (policy)            │
│               │  (Policy)  │                                │
│               └────────────┘                                │
│                                                              │
│     State ──→ ┌────────────┐                                │
│               │   Critic   │ ──→ Value (evaluation)         │
│               │  (Value)   │                                │
│               └────────────┘                                │
│                                                              │
│  Actor: Decides which action to take                        │
│  Critic: Evaluates how good the action was                  │
│  Training: Critic's evaluation guides actor's updates       │
└─────────────────────────────────────────────────────────────┘

Variants:
- A2C: Advantage Actor-Critic
- A3C: Asynchronous A3C
- PPO: Proximal Policy Optimization
- SAC: Soft Actor-Critic
```

---

## ⚖️ AI Ethics & Safety

### Bias in AI

```
Types of Bias:
┌─────────────────────────────────────────────────────────────┐
│ Historical Bias:                                            │
│ - Bias present in training data from society               │
│                                                              │
│ Representation Bias:                                        │
│ - Underrepresentation of certain groups                    │
│                                                              │
│ Measurement Bias:                                           │
│ - Flawed metrics or proxy variables                        │
│                                                              │
│ Aggregation Bias:                                           │
│ - One-size-fits-all models for diverse groups              │
│                                                              │
│ Evaluation Bias:                                            │
│ - Testing on non-representative benchmarks                 │
└─────────────────────────────────────────────────────────────┘

Mitigation Strategies:
- Diverse and representative training data
- Bias audits and testing
- Fairness constraints in optimization
- Continuous monitoring in production
```

### Fairness Definitions

| Definition | Description |
|------------|-------------|
| **Demographic Parity** | Equal positive rate across groups |
| **Equalized Odds** | Equal TPR and FPR across groups |
| **Individual Fairness** | Similar individuals get similar predictions |
| **Counterfactual Fairness** | Prediction unchanged if protected attribute changed |

### Explainable AI (XAI)

```
Methods:
┌─────────────────────────────────────────────────────────────┐
│ LIME (Local Interpretable Model-agnostic Explanations):     │
│ - Create interpretable local approximation                  │
│ - Works for any model                                       │
│                                                              │
│ SHAP (SHapley Additive exPlanations):                       │
│ - Based on game theory (Shapley values)                     │
│ - Feature contribution to prediction                        │
│                                                              │
│ Attention Visualization:                                    │
│ - Show which parts of input model attends to               │
│                                                              │
│ Saliency Maps:                                              │
│ - Gradient-based pixel importance for images               │
│                                                              │
│ Concept Activation Vectors (CAV):                           │
│ - Test sensitivity to human concepts                       │
└─────────────────────────────────────────────────────────────┘
```

### AI Alignment

```
Key Challenges:

1. Specification Problem:
   - Difficulty in precisely defining what we want
   - Goodhart's Law: "When a measure becomes a target, 
     it ceases to be a good measure"

2. Robustness Problem:
   - AI behaving correctly in distribution
   - Failing on edge cases or distribution shift

3. Assurance Problem:
   - How do we verify AI is aligned?
   - Can we trust AI's explanations?

4. Deception Risk:
   - Sufficiently capable AI might deceive evaluators
   - Instrumental convergence concerns

Approaches:
- Constitutional AI
- Debate (AI arguing with itself)
- Recursive reward modeling
- Interpretability research
```

### Responsible AI Practices

```
Framework:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  1. TRANSPARENCY                                            │
│     - Document model cards                                  │
│     - Publish limitations                                   │
│     - Clear usage guidelines                                │
│                                                              │
│  2. ACCOUNTABILITY                                          │
│     - Clear ownership                                       │
│     - Audit trails                                          │
│     - Incident response plans                               │
│                                                              │
│  3. PRIVACY                                                 │
│     - Data minimization                                     │
│     - Consent mechanisms                                    │
│     - Differential privacy                                  │
│                                                              │
│  4. SECURITY                                                │
│     - Adversarial robustness                               │
│     - Access controls                                       │
│     - Regular security audits                              │
│                                                              │
│  5. HUMAN OVERSIGHT                                         │
│     - Human-in-the-loop for critical decisions             │
│     - Appeal mechanisms                                     │
│     - Regular human review                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

<div align="center">

## 📚 Continue Learning

| Section | Link |
|---------|------|
| 💻 Code Examples | [Browse Code →](../code/README.md) |
| 🔗 Resources | [Browse Resources →](../resources/README.md) |
| 📋 Cheatsheets | [Browse Cheatsheets →](../cheatsheets/README.md) |
| 📖 Glossary | [Browse Glossary →](../glossary/README.md) |

---

[← Back to Main](../README.md)

</div>
