# 🧠 Deep Learning

## Table of Contents
- [Introduction](#introduction)
- [Neural Network Basics](#neural-network-basics)
- [Backpropagation](#backpropagation)
- [Activation Functions](#activation-functions)
- [Optimization](#optimization)
- [Regularization](#regularization)
- [Architectures](#architectures)
- [Practical Tips](#practical-tips)

---

## Introduction

Deep Learning is a subset of machine learning based on artificial neural networks with multiple layers (hence "deep"). These networks can automatically learn hierarchical representations of data.

```
┌────────────────────────────────────────────────────────────────────────┐
│                     AI → ML → DEEP LEARNING                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    ╭───────────────────────────────────────────────────────────────╮   │
│    │                  ARTIFICIAL INTELLIGENCE                       │   │
│    │    ╭───────────────────────────────────────────────────╮      │   │
│    │    │              MACHINE LEARNING                      │      │   │
│    │    │    ╭───────────────────────────────────────╮      │      │   │
│    │    │    │            DEEP LEARNING               │      │      │   │
│    │    │    │  • Neural Networks                     │      │      │   │
│    │    │    │  • CNNs, RNNs, Transformers           │      │      │   │
│    │    │    │  • Representation Learning            │      │      │   │
│    │    │    ╰───────────────────────────────────────╯      │      │   │
│    │    ╰───────────────────────────────────────────────────╯      │   │
│    ╰───────────────────────────────────────────────────────────────╯   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Neural Network Basics

### The Perceptron

The simplest neural network unit:

```
        x₁ ──[w₁]──┐
                    │
        x₂ ──[w₂]──┼──[Σ]──[f]── y
                    │
        x₃ ──[w₃]──┘
                    │
             b ─────┘
```

$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(W^T X + b)$$

### Multi-Layer Perceptron (MLP)

```
    INPUT          HIDDEN          HIDDEN          OUTPUT
    LAYER          LAYER 1         LAYER 2         LAYER
    
     (x₁)──────────(h₁₁)──────────(h₂₁)
        ╲        ╱    ╲        ╱    ╲        ╲
         ╲      ╱      ╲      ╱      ╲        (y₁)
     (x₂)──╲──╱────────(h₁₂)──────────(h₂₂)───╱
            ╲╱              ╲╱            ╲  ╱
            ╱╲              ╱╲            ╱╲
     (x₃)──╱──╲────────(h₁₃)──────────(h₂₃)───╲
         ╱      ╲      ╱      ╲      ╱        (y₂)
        ╱        ╲    ╱        ╲    ╱
     (x₄)──────────(h₁₄)──────────(h₂₄)
```

### Forward Pass

For layer $l$:
$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

Where:
- $W^{[l]}$ = weight matrix for layer $l$
- $A^{[l]}$ = activations of layer $l$
- $g^{[l]}$ = activation function for layer $l$

---

## Backpropagation

The algorithm for training neural networks by computing gradients.

### Chain Rule

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Backward Pass Algorithm

```
┌────────────────────────────────────────────────────────────────────────┐
│                    BACKPROPAGATION FLOW                                 │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    FORWARD PASS ───────────────────────────────────────────────────►   │
│                                                                         │
│    Input    Layer 1    Layer 2    Layer 3    Output    Loss            │
│      X   →   W₁,b₁  →  W₂,b₂  →  W₃,b₃  →    ŷ    →    L             │
│                                                                         │
│    ◄─────────────────────────────────────────────────── BACKWARD PASS  │
│                                                                         │
│     ∂L       ∂L        ∂L        ∂L         ∂L                         │
│    ────     ────      ────      ────       ────                        │
│     ∂X      ∂W₁       ∂W₂       ∂W₃        ∂ŷ                         │
└────────────────────────────────────────────────────────────────────────┘
```

For output layer:
$$\delta^{[L]} = \frac{\partial L}{\partial A^{[L]}} \odot g'^{[L]}(Z^{[L]})$$

For hidden layers:
$$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot g'^{[l]}(Z^{[l]})$$

Weight gradients:
$$\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (A^{[l-1]})^T$$

---

## Activation Functions

| Function | Formula | Derivative | Range |
|----------|---------|------------|-------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | (0, 1) |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | (-1, 1) |
| **ReLU** | $\max(0, x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | [0, ∞) |
| **Leaky ReLU** | $\max(0.01x, x)$ | $\begin{cases} 1 & x > 0 \\ 0.01 & x \leq 0 \end{cases}$ | (-∞, ∞) |
| **ELU** | $\begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$ | $\begin{cases} 1 & x > 0 \\ f(x) + \alpha & x \leq 0 \end{cases}$ | (-α, ∞) |
| **GELU** | $x \cdot \Phi(x)$ | Complex | (-∞, ∞) |
| **Swish** | $x \cdot \sigma(x)$ | Complex | (-∞, ∞) |

### Visualization

```
         Sigmoid              ReLU              Leaky ReLU
     1 ┤    ╭────────      ┤        ╱          ┤        ╱
       │   ╱               │       ╱           │       ╱
   0.5 ┤──╯                │      ╱            │      ╱
       │                   │     ╱             │    ╱
     0 ┼────────────     0 ┼────╱───────     0 ┼───╱──────
       │                   │                   │  ╱
       └─────┬─────        └─────┬─────        └─╱───┬─────
            -4  4               -4  4              -4  4
```

### When to Use What

- **ReLU**: Default choice for hidden layers
- **Leaky ReLU/ELU**: When facing dying ReLU problem
- **GELU/Swish**: Modern architectures (Transformers)
- **Sigmoid**: Output layer for binary classification
- **Softmax**: Output layer for multi-class classification

---

## Optimization

### Gradient Descent Variants

#### 1. Vanilla Gradient Descent
$$\theta = \theta - \alpha \nabla_\theta L(\theta)$$

#### 2. Momentum
$$v_t = \gamma v_{t-1} + \alpha \nabla_\theta L(\theta)$$
$$\theta = \theta - v_t$$

#### 3. RMSprop
$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g_t^2$$
$$\theta = \theta - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

#### 4. Adam (Recommended)
Combines momentum and RMSprop:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta = \theta - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### Learning Rate Schedules

```
Constant        Step Decay       Exponential      Cosine Annealing
   │                │                │                   │
α  ┼────────     α  ┼──┐             │ ╲              α  ╭────╮
   │                │  └──┐        α ┼  ╲               ╱    ╲
   │                │     └──      │   ╲──            ╱       ╲
   └───────►       └───────►       └──────►         └─────────►
     epochs          epochs          epochs            epochs
```

---

## Regularization

### 1. L1 and L2 Regularization

**L2 (Ridge/Weight Decay):**
$$L_{total} = L_{original} + \lambda \sum_i w_i^2$$

**L1 (Lasso):**
$$L_{total} = L_{original} + \lambda \sum_i |w_i|$$

### 2. Dropout

Randomly set activations to zero during training:

```
   Training:                    Testing:
   
   [1.0] ✓                      [1.0]
   [0.5] ✗ → 0                  [0.5]
   [0.8] ✓          →           [0.8]  × (1-p)
   [0.3] ✗ → 0                  [0.3]
   [0.7] ✓                      [0.7]
```

### 3. Batch Normalization

Normalize activations within each mini-batch:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

**Benefits:**
- Faster training
- Higher learning rates
- Reduces internal covariate shift

### 4. Early Stopping

```
Loss
  │
  │  ╲ Training Loss
  │   ╲
  │    ╲    
  │     ╲_______________
  │          ╱
  │         ╱ Validation Loss
  │   ╲    ╱
  │    ╲__╱
  │       ↑
  │    STOP HERE
  └─────────────────────► Epochs
```

---

## Architectures

### Convolutional Neural Networks (CNNs)

```
┌───────────────────────────────────────────────────────────────────┐
│                    CNN ARCHITECTURE                                │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│   INPUT     CONV    POOL    CONV    POOL    FLATTEN    FC         │
│                                                                    │
│  ┌─────┐   ┌─────┐  ┌───┐  ┌─────┐  ┌───┐   ┌─────┐  ┌─────┐     │
│  │     │   │     │  │   │  │     │  │   │   │     │  │     │     │
│  │Image│ → │Conv │→ │Max│→ │Conv │→ │Max│ → │  F  │→ │ FC  │ → y │
│  │     │   │+ReLU│  │Pool│  │+ReLU│  │Pool│  │  L  │  │ +  │     │
│  │     │   │     │  │   │  │     │  │   │   │  A  │  │Soft│     │
│  └─────┘   └─────┘  └───┘  └─────┘  └───┘   │  T  │  │max │     │
│  224×224   222×222  111×111                  └─────┘  └─────┘     │
│   ×3         ×32     ×32                       4096    1000       │
└───────────────────────────────────────────────────────────────────┘
```

### Recurrent Neural Networks (RNNs)

```
         ┌─────────────────────────────────────────────────┐
         │                   UNROLLED RNN                   │
         ├─────────────────────────────────────────────────┤
         │                                                  │
         │    h₀    h₁    h₂    h₃                         │
         │     │     │     │     │                         │
         │     ▼     ▼     ▼     ▼                         │
         │   ┌───┐ ┌───┐ ┌───┐ ┌───┐                       │
         │   │RNN│→│RNN│→│RNN│→│RNN│→ ...                  │
         │   └───┘ └───┘ └───┘ └───┘                       │
         │     ▲     ▲     ▲     ▲                         │
         │     │     │     │     │                         │
         │    x₀    x₁    x₂    x₃                         │
         │   "The" "cat" "sat" "on"                        │
         └─────────────────────────────────────────────────┘
```

### LSTM (Long Short-Term Memory)

```
                    ┌──────────────────────────────┐
                    │         LSTM CELL            │
                    ├──────────────────────────────┤
                    │                              │
              c_{t-1} ────────×─────────+────────────→ c_t
                    │         │    ↑    │          │
                    │         │  ┌───┐  │          │
                    │         │  │ σ │ tanh       │
                    │         │  └─┬─┘  │          │
                    │         │    │    │          │
              h_{t-1} ─────┬──┴────┴────┴──────────┬──→ h_t
                    │      │                       │  │
                    │   ┌──┴──┐               ┌──┴──┐│
                    │   │  σ  │  σ   tanh     │ tanh││
                    │   │ f_t │ i_t   g_t     │  o_t││
                    │   └──┬──┘               └──┬──┘│
                    │      └───────┬───────────┘   │
                    │              │                │
                    │             x_t              │
                    └──────────────────────────────┘

Gates:
- Forget gate (f): What to forget from cell state
- Input gate (i): What new info to store
- Output gate (o): What to output
```

---

## Practical Tips

### 1. Weight Initialization
- **Xavier/Glorot**: For sigmoid/tanh - $W \sim U\left[-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}\right]$
- **He**: For ReLU - $W \sim N\left(0, \sqrt{\frac{2}{n_{in}}}\right)$

### 2. Gradient Checking
Verify backprop implementation:
$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$

### 3. Common Issues

| Problem | Symptoms | Solutions |
|---------|----------|-----------|
| **Vanishing Gradients** | Slow/no learning in early layers | ReLU, residual connections, LSTM |
| **Exploding Gradients** | NaN losses, unstable training | Gradient clipping, lower LR |
| **Overfitting** | Train loss ↓, Val loss ↑ | Dropout, regularization, more data |
| **Underfitting** | Both losses high | Larger model, train longer |

### 4. Debugging Neural Networks

1. **Start small**: Overfit on a tiny dataset first
2. **Verify loss**: Check initial loss is reasonable
3. **Monitor gradients**: Look for dead neurons
4. **Visualize**: Training curves, activations, weights

---

## Resources

- 📚 **Book**: "Deep Learning" by Goodfellow, Bengio, and Courville
- 🎓 **Course**: Stanford CS231n - CNNs for Visual Recognition
- 🎓 **Course**: deeplearning.ai by Andrew Ng
- 📄 **Paper**: "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet)

---

🌐 [Back to Notes](README.md) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
