# 🎮 Reinforcement Learning

## Table of Contents
- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
- [Markov Decision Processes](#markov-decision-processes)
- [Value-Based Methods](#value-based-methods)
- [Policy-Based Methods](#policy-based-methods)
- [Actor-Critic Methods](#actor-critic-methods)
- [Deep Reinforcement Learning](#deep-reinforcement-learning)
- [Applications](#applications)

---

## Introduction

Reinforcement Learning (RL) is learning through interaction with an environment to maximize cumulative reward.

```
┌────────────────────────────────────────────────────────────────────────┐
│                    RL PARADIGM                                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│              ┌─────────────────────────────────────────┐               │
│              │            ENVIRONMENT                   │               │
│              └──────────────┬──────────────────────────┘               │
│                             │                                           │
│                    state s, reward r                                    │
│                             │                                           │
│                             ▼                                           │
│              ┌─────────────────────────────────────────┐               │
│              │              AGENT                       │               │
│              │                                          │               │
│              │  Observes state → Selects action         │               │
│              │  Receives reward → Updates policy        │               │
│              │                                          │               │
│              └──────────────┬──────────────────────────┘               │
│                             │                                           │
│                         action a                                        │
│                             │                                           │
│                             ▼                                           │
│              ┌─────────────────────────────────────────┐               │
│              │            ENVIRONMENT                   │               │
│              └─────────────────────────────────────────┘               │
│                                                                         │
│   Goal: Learn policy π that maximizes expected cumulative reward       │
└────────────────────────────────────────────────────────────────────────┘
```

### RL vs Other Learning Paradigms

| Paradigm | Supervision | Feedback | Examples |
|----------|-------------|----------|----------|
| **Supervised** | Full labels | Immediate, exact | Classification |
| **Unsupervised** | No labels | None | Clustering |
| **Reinforcement** | Reward signal | Delayed, scalar | Game playing |

---

## Core Concepts

### Key Terminology

| Term | Symbol | Description |
|------|--------|-------------|
| **State** | $s$ | Current situation of the environment |
| **Action** | $a$ | Choice made by the agent |
| **Reward** | $r$ | Immediate feedback signal |
| **Policy** | $\pi(a\|s)$ | Strategy mapping states to actions |
| **Value** | $V(s)$ | Expected cumulative reward from state |
| **Q-Value** | $Q(s,a)$ | Expected cumulative reward from state-action pair |
| **Return** | $G_t$ | Total discounted reward from time $t$ |

### The RL Loop

```
t=0        t=1        t=2        t=3
 │          │          │          │
 s₀ ──a₀──► s₁ ──a₁──► s₂ ──a₂──► s₃ ──► ...
 │    │     │    │     │    │     │
 └──r₁──┘   └──r₂──┘   └──r₃──┘   ...

Return: G₀ = r₁ + γr₂ + γ²r₃ + ...
        where γ ∈ [0, 1] is discount factor
```

### Exploration vs Exploitation

```
┌────────────────────────────────────────────────────────────────────────┐
│                    EXPLORATION-EXPLOITATION TRADEOFF                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   EXPLOITATION                    EXPLORATION                          │
│   "Use what you know"            "Try new things"                      │
│                                                                         │
│   Go to your favorite            Try a new restaurant                  │
│   restaurant                                                            │
│                                                                         │
│   Guaranteed good meal           Might find something better           │
│   but no improvement             but might be disappointing            │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────    │
│                                                                         │
│   Common strategies:                                                   │
│   • ε-greedy: Random action with probability ε                         │
│   • Softmax: Sample from action distribution                           │
│   • UCB: Upper Confidence Bound                                        │
│   • Thompson Sampling: Sample from posterior                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Markov Decision Processes

### Definition

An MDP is defined by tuple $(S, A, P, R, \gamma)$:
- $S$: Set of states
- $A$: Set of actions
- $P(s'|s, a)$: Transition probability
- $R(s, a, s')$: Reward function
- $\gamma$: Discount factor

### Markov Property

$$P(s_{t+1} | s_t, a_t, s_{t-1}, ..., s_0) = P(s_{t+1} | s_t, a_t)$$

"The future depends only on the present, not the past"

### Bellman Equations

**Value Function:**
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s\right]$$

**Bellman Expectation Equation:**
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**Bellman Optimality Equation:**
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

```
┌────────────────────────────────────────────────────────────────────────┐
│                    BELLMAN BACKUP DIAGRAM                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         V(s)                                            │
│                          │                                              │
│          ┌───────────────┼───────────────┐                             │
│          │               │               │                             │
│          ▼               ▼               ▼                             │
│         a₁              a₂              a₃        (policy chooses)     │
│          │               │               │                             │
│     ┌────┼────┐     ┌────┼────┐     ┌────┼────┐                        │
│     │    │    │     │    │    │     │    │    │                        │
│     ▼    ▼    ▼     ▼    ▼    ▼     ▼    ▼    ▼                        │
│    s'₁  s'₂  s'₃   s'₁  s'₂  s'₃   s'₁  s'₂  s'₃  (transition probs)  │
│     │    │    │     │    │    │     │    │    │                        │
│    V(s'₁) ...                                       (recursive values) │
│                                                                         │
│   V(s) = Σₐ π(a|s) Σₛ' P(s'|s,a) [R + γV(s')]                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Value-Based Methods

### Q-Learning

Learn action-value function directly (model-free, off-policy):

$$Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

```
Algorithm: Q-Learning
─────────────────────────────────────────────────────────
Initialize Q(s, a) arbitrarily
For each episode:
    Initialize s
    For each step:
        Choose a from s using ε-greedy from Q
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        s ← s'
    Until s is terminal
```

### SARSA

On-policy TD control:

$$Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma Q(s', a') - Q(s, a)]$$

```
Q-Learning vs SARSA:

Q-Learning (off-policy):     SARSA (on-policy):
Update uses max Q(s',a')     Update uses actual a' taken
                             
Q(s,a) + α[r + γ max Q - Q]  Q(s,a) + α[r + γ Q(s',a') - Q]
              ↑                              ↑
         Best action                   Action actually taken
```

### TD Learning

**Temporal Difference**: Bootstrap from current estimates

$$V(s) \leftarrow V(s) + \alpha[\underbrace{r + \gamma V(s')}_{\text{TD target}} - V(s)]$$

```
                TD error = r + γV(s') - V(s)
                            ↑
                    Bootstrapped estimate
                    (using current V)

Monte Carlo:     Wait until episode ends, use actual return
TD(0):           Use immediate reward + next state estimate
TD(λ):           Blend of MC and TD with eligibility traces
```

---

## Policy-Based Methods

### Policy Gradient

Directly optimize the policy:

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

**Policy Gradient Theorem:**
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a)\right]$$

### REINFORCE

```
Algorithm: REINFORCE (Monte Carlo Policy Gradient)
─────────────────────────────────────────────────────────
Initialize policy parameters θ
For each episode:
    Generate episode: s₀, a₀, r₁, s₁, a₁, r₂, ..., sₜ
    For t = 0, 1, ..., T:
        G ← Σ_{k=t}^{T} γ^{k-t} r_{k+1}    (return from step t)
        θ ← θ + α G ∇_θ log π_θ(aₜ|sₜ)
```

### Advantages of Policy Gradient

| Aspect | Value-Based | Policy-Based |
|--------|-------------|--------------|
| Action space | Discrete (usually) | Continuous ✓ |
| Stochastic policies | Hard | Natural ✓ |
| Convergence | Can oscillate | Smoother ✓ |
| Sample efficiency | Better | Lower |

---

## Actor-Critic Methods

Combine policy gradient (actor) with value function (critic):

```
┌────────────────────────────────────────────────────────────────────────┐
│                    ACTOR-CRITIC ARCHITECTURE                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    Environment                                          │
│                        │                                                │
│              ┌─────────┴─────────┐                                     │
│              │  state s, reward r │                                    │
│              └─────────┬─────────┘                                     │
│                        │                                                │
│          ┌─────────────┴─────────────┐                                 │
│          │                           │                                 │
│          ▼                           ▼                                 │
│    ┌───────────┐              ┌───────────┐                            │
│    │   ACTOR   │              │  CRITIC   │                            │
│    │   π(a|s)  │              │   V(s)    │                            │
│    │           │              │           │                            │
│    │  Policy   │◄─────────────│  Value    │                            │
│    │  Network  │  TD error    │  Network  │                            │
│    └─────┬─────┘              └───────────┘                            │
│          │                                                              │
│          ▼                                                              │
│       action a                                                          │
│                                                                         │
│   Actor: Updates policy to maximize expected value                     │
│   Critic: Evaluates how good the actor's actions are                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Advantage Function

$$A(s, a) = Q(s, a) - V(s)$$

"How much better is action $a$ compared to average?"

### A2C/A3C

**Advantage Actor-Critic:**
$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) A(s, a)]$$

**A3C**: Asynchronous variant with parallel actors

### PPO (Proximal Policy Optimization)

Most popular modern algorithm:

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)\right]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

```
PPO Key Ideas:
1. Clipped objective prevents too large policy updates
2. Multiple epochs per batch (sample efficient)
3. Stable training without complex tuning
```

---

## Deep Reinforcement Learning

### DQN (Deep Q-Network)

Q-Learning + Deep Neural Networks:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    DQN ARCHITECTURE                                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   State (4 frames)                                                     │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────┐                                                          │
│   │  Conv   │  84×84×4 → 32 filters                                    │
│   └────┬────┘                                                          │
│        │                                                                │
│   ┌────▼────┐                                                          │
│   │  Conv   │  → 64 filters                                            │
│   └────┬────┘                                                          │
│        │                                                                │
│   ┌────▼────┐                                                          │
│   │  Conv   │  → 64 filters                                            │
│   └────┬────┘                                                          │
│        │                                                                │
│   ┌────▼────┐                                                          │
│   │   FC    │  512 units                                               │
│   └────┬────┘                                                          │
│        │                                                                │
│   ┌────▼────┐                                                          │
│   │   FC    │  |A| outputs (Q-value per action)                        │
│   └─────────┘                                                          │
│                                                                         │
│   Key innovations:                                                     │
│   • Experience Replay: Store and sample past transitions               │
│   • Target Network: Separate network for stable targets                │
└────────────────────────────────────────────────────────────────────────┘
```

### DQN Improvements

| Improvement | Description |
|-------------|-------------|
| **Double DQN** | Use online network to select, target to evaluate |
| **Dueling DQN** | Separate value and advantage streams |
| **Prioritized Replay** | Sample important transitions more often |
| **Noisy Nets** | Parametric exploration |
| **Rainbow** | Combines all improvements |

### Policy Gradient with Deep Networks

```
TRPO → PPO → SAC

TRPO: Trust Region Policy Optimization
      Complex constraint optimization

PPO: Proximal Policy Optimization  
     Simpler clipped objective
     
SAC: Soft Actor-Critic
     Maximum entropy RL
     Better exploration
```

---

## Applications

### Games

```
┌────────────────────────────────────────────────────────────────────────┐
│                    RL IN GAMES                                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   2013: DQN           Atari games (superhuman on many)                 │
│   2016: AlphaGo       Go (beat world champion)                         │
│   2017: AlphaZero     Chess, Shogi, Go (from scratch)                  │
│   2019: AlphaStar     StarCraft II (Grandmaster level)                 │
│   2019: OpenAI Five   Dota 2 (beat world champions)                    │
│   2022: Cicero        Diplomacy (human-level negotiation)              │
└────────────────────────────────────────────────────────────────────────┘
```

### Real-World Applications

| Domain | Application |
|--------|-------------|
| **Robotics** | Robot control, manipulation |
| **Autonomous Vehicles** | Navigation, decision making |
| **Recommendation** | Personalized content |
| **Trading** | Portfolio optimization |
| **Healthcare** | Treatment optimization |
| **LLMs** | RLHF for alignment |

### RLHF for LLMs

```
Pre-training → SFT → Reward Model → PPO Fine-tuning
                           │
                           └── Human preferences
                               "Which response is better?"
```

---

## Summary: Algorithm Selection

```
┌────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE WHAT                                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Discrete Actions + Low Dimensional State:                            │
│   → Q-Learning, DQN                                                    │
│                                                                         │
│   Continuous Actions:                                                  │
│   → PPO, SAC, DDPG                                                     │
│                                                                         │
│   Simple Environment + Fast Iteration:                                 │
│   → PPO (reliable, easy to tune)                                       │
│                                                                         │
│   Sample Efficiency Critical:                                          │
│   → SAC, model-based methods                                           │
│                                                                         │
│   Multi-Agent:                                                         │
│   → MARL algorithms (MAPPO, QMIX)                                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Resources

- 📚 **Book**: "Reinforcement Learning" by Sutton & Barto (free online)
- 🎓 **Course**: David Silver's RL Course (DeepMind)
- 🎓 **Course**: UC Berkeley CS285 - Deep RL
- 📄 **Paper**: "Playing Atari with Deep RL" (DQN)
- 📄 **Paper**: "Proximal Policy Optimization Algorithms" (PPO)

---

🌐 [Back to Notes](README.md) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
