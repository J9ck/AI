# 📚 AI/ML Glossary

A comprehensive dictionary of AI and Machine Learning terminology.
# 📖 AI/ML Glossary

> Comprehensive A-Z terminology reference for Artificial Intelligence and Machine Learning.

[← Back to Main](../README.md)

---

## 📋 Quick Navigation

[A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [H](#h) | [I](#i) | [J](#j) | [K](#k) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [Q](#q) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w) | [X](#x) | [Y](#y) | [Z](#z)

---

## A

**Activation Function**
A mathematical function applied to a neuron's output to introduce non-linearity. Common examples: ReLU, Sigmoid, Tanh, Softmax.

**Adam (Adaptive Moment Estimation)**
An optimization algorithm combining momentum and RMSprop. Default optimizer for many deep learning tasks.

**Attention Mechanism**
A technique that allows models to focus on relevant parts of the input when producing output. Foundation of Transformer architecture.

**Autoencoder**
A neural network that learns to compress data into a lower-dimensional representation and reconstruct it. Used for dimensionality reduction and generative modeling.

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**
A metric for evaluating binary classification models. Measures the model's ability to distinguish between classes.
**Accuracy**
The proportion of correct predictions (both true positives and true negatives) among the total number of predictions. While intuitive, it can be misleading for imbalanced datasets.

**Activation Function**
A mathematical function applied to the output of a neuron that introduces non-linearity into the network. Common examples include ReLU, Sigmoid, Tanh, and Softmax.

**Adam (Adaptive Moment Estimation)**
An optimization algorithm that combines the benefits of AdaGrad and RMSprop. It maintains adaptive learning rates for each parameter using estimates of first and second moments of gradients.

**Adversarial Examples**
Inputs that have been intentionally modified to cause a machine learning model to make mistakes. Often imperceptible to humans but can fool neural networks.

**Agent**
In reinforcement learning, an entity that learns to make decisions by interacting with an environment to maximize cumulative rewards.

**AI Alignment**
The challenge of ensuring that artificial intelligence systems behave in accordance with human values and intentions, especially as systems become more capable.

**Anomaly Detection**
The identification of rare items, events, or observations that differ significantly from the majority of the data. Also called outlier detection.

**Attention Mechanism**
A neural network component that allows models to focus on relevant parts of the input when producing an output. Foundation of transformer architectures.

**Augmentation (Data)**
Techniques to artificially increase the size of a training dataset by creating modified versions of existing data (e.g., rotating images, adding noise).

**Autoencoder**
A neural network architecture that learns to compress data into a lower-dimensional representation and then reconstruct it. Used for dimensionality reduction, denoising, and generative modeling.

**AutoML**
Automated Machine Learning - tools and techniques that automate the process of applying machine learning, including feature engineering, model selection, and hyperparameter tuning.

---

## B

**Backpropagation**
The algorithm used to calculate gradients for training neural networks. Propagates error backward through the network to update weights.

**Batch Normalization**
A technique to normalize layer inputs during training, improving training speed and stability.

**Batch Size**
The number of training samples processed before updating model weights.

**BERT (Bidirectional Encoder Representations from Transformers)**
A pre-trained language model using bidirectional context. Foundation for many NLP applications.

**Bias (ML)**
1. A parameter in a model (intercept term)
2. Systematic error in predictions
3. In bias-variance tradeoff: error from overly simple models

**Bias-Variance Tradeoff**
The balance between underfitting (high bias) and overfitting (high variance).
The algorithm used to calculate gradients of the loss function with respect to weights in a neural network, enabling training through gradient descent.

**Batch**
A subset of training examples used in one iteration of model training. Batch size affects training speed, memory usage, and convergence.

**Batch Normalization**
A technique that normalizes the inputs to each layer, reducing internal covariate shift and enabling faster, more stable training.

**Bayes' Theorem**
A fundamental theorem in probability that describes how to update beliefs based on new evidence: P(A|B) = P(B|A) × P(A) / P(B).

**BERT (Bidirectional Encoder Representations from Transformers)**
A language model pre-trained on unlabeled text using bidirectional context. Revolutionized NLP by enabling transfer learning for many tasks.

**Bias (Statistical)**
The difference between a model's expected predictions and the true values. High bias leads to underfitting.

**Bias (ML Fairness)**
Systematic prejudice in training data or algorithms that leads to unfair outcomes for certain groups.

**Binary Classification**
A classification task where the target variable has exactly two possible classes (e.g., spam/not spam).

**Boosting**
An ensemble method that trains models sequentially, with each new model focusing on correcting the errors of previous ones. Examples include AdaBoost, XGBoost, and LightGBM.

---

## C

**Chain-of-Thought (CoT)**
A prompting technique that encourages LLMs to break down reasoning into steps.

**Classification**
Predicting a categorical label for an input. Binary (2 classes) or multiclass (3+ classes).

**Clustering**
Unsupervised learning technique to group similar data points together.

**CNN (Convolutional Neural Network)**
A neural network architecture using convolutional layers, primarily for image processing.

**Constitutional AI**
An approach to AI alignment where models follow principles outlined in a "constitution."

**Convolution**
A mathematical operation that slides a filter over input data to detect features.

**Cross-Entropy Loss**
A loss function commonly used for classification tasks. Measures difference between predicted probabilities and true labels.

**Cross-Validation**
A technique to evaluate model performance by training on different subsets of data.
A prompting technique where models are encouraged to break down complex reasoning into intermediate steps, improving performance on multi-step problems.

**ChatGPT**
OpenAI's conversational AI model based on GPT architecture, fine-tuned with RLHF to follow instructions and engage in dialogue.

**Classification**
A supervised learning task where the goal is to predict which category or class an input belongs to.

**Clustering**
An unsupervised learning task that groups similar data points together based on their features without predefined labels.

**CNN (Convolutional Neural Network)**
A neural network architecture specialized for processing grid-like data (images). Uses convolutional layers to automatically learn spatial hierarchies of features.

**Confusion Matrix**
A table showing the performance of a classification model: True Positives, True Negatives, False Positives, and False Negatives.

**Constitutional AI**
An approach to AI alignment where the model is trained to follow a set of principles or "constitution" that guides its behavior.

**Contrastive Learning**
A self-supervised learning approach that learns representations by bringing similar samples closer and pushing dissimilar samples apart in embedding space.

**Convolution**
A mathematical operation that applies a filter/kernel across input data to extract features. Fundamental operation in CNNs.

**Cross-Entropy Loss**
A loss function commonly used for classification tasks that measures the difference between predicted probability distributions and actual labels.

**Cross-Validation**
A technique for assessing model performance by splitting data into multiple folds and training/testing on different combinations.

---

## D

**Data Augmentation**
Artificially increasing training data by applying transformations (rotation, flipping, etc.).

**Deep Learning**
Machine learning using neural networks with multiple layers.

**Diffusion Model**
A generative model that learns to reverse a gradual noising process. Used in image generation (Stable Diffusion, DALL-E).

**Discriminator**
In GANs, the network that distinguishes between real and generated data.

**Dropout**
A regularization technique that randomly sets neurons to zero during training.
See Augmentation (Data).

**Decision Tree**
A supervised learning algorithm that makes predictions by learning decision rules inferred from features, creating a tree-like structure.

**Deep Learning**
A subset of machine learning using neural networks with multiple layers (deep architectures) to learn hierarchical representations of data.

**Diffusion Models**
Generative models that learn to create data by gradually denoising random noise. Powers models like Stable Diffusion and DALL-E.

**Dimensionality Reduction**
Techniques to reduce the number of features in a dataset while preserving important information. Examples include PCA and t-SNE.

**Discriminator**
In GANs, the network that tries to distinguish between real and generated (fake) samples.

**Dropout**
A regularization technique that randomly sets a fraction of neuron outputs to zero during training, preventing overfitting.

**DQN (Deep Q-Network)**
A reinforcement learning algorithm that combines Q-learning with deep neural networks, enabling RL in high-dimensional state spaces.

---

## E

**Embedding**
A dense vector representation of discrete data (words, users, items). Captures semantic relationships.

**Encoder-Decoder**
An architecture where an encoder processes input into a representation, and a decoder generates output from it.

**Ensemble**
Combining multiple models to improve predictions. Examples: Random Forest, XGBoost.

**Epoch**
One complete pass through the entire training dataset.

**Explainability (XAI)**
Techniques to understand and interpret model decisions.
A learned representation of discrete items (words, items, users) as dense vectors in a continuous space where similar items are close together.

**Encoder**
A network component that compresses input data into a latent representation. Used in autoencoders, sequence-to-sequence models, and transformers.

**Ensemble Methods**
Techniques that combine multiple models to produce better predictions than any single model. Includes bagging, boosting, and stacking.

**Epoch**
One complete pass through the entire training dataset during model training.

**Explainability (XAI)**
The degree to which AI system decisions can be understood by humans. Techniques include LIME, SHAP, and attention visualization.

---

## F

**Feature**
An individual measurable property used as input to a model.

**Feature Engineering**
The process of creating, selecting, and transforming features to improve model performance.

**Few-Shot Learning**
Learning from very few examples. In LLMs, providing examples in the prompt.

**Fine-Tuning**
Adapting a pre-trained model to a specific task by training on task-specific data.

**F1 Score**
The harmonic mean of precision and recall. Useful for imbalanced datasets.
**F1 Score**
The harmonic mean of precision and recall, providing a single metric that balances both concerns. Useful for imbalanced datasets.

**Feature**
An individual measurable property or characteristic of the data used as input to a model.

**Feature Engineering**
The process of using domain knowledge to create, select, or transform features to improve model performance.

**Feature Extraction**
Automatically learning useful representations from raw data, often using neural networks.

**Feedforward Network**
A neural network where information flows in one direction from input to output, without cycles.

**Few-Shot Learning**
The ability of a model to learn from only a few examples, often through careful prompting or meta-learning.

**Fine-Tuning**
The process of taking a pre-trained model and further training it on a specific task or domain with additional data.

**FSDP (Fully Sharded Data Parallel)**
A distributed training technique that shards model parameters across GPUs to enable training of very large models.

---

## G

**GAN (Generative Adversarial Network)**
A generative model with two networks (generator and discriminator) competing against each other.

**Generative AI**
AI systems that create new content (text, images, audio, video).

**Generator**
In GANs, the network that creates synthetic data.

**Gradient**
The vector of partial derivatives. Points in the direction of steepest increase of a function.

**Gradient Descent**
An optimization algorithm that iteratively adjusts parameters in the direction of negative gradient.

**GPT (Generative Pre-trained Transformer)**
A family of autoregressive language models by OpenAI.
A generative model architecture where two networks (generator and discriminator) compete: one creates fake data, the other tries to detect it.

**Generalization**
A model's ability to perform well on new, unseen data rather than just memorizing the training data.

**Generative AI**
AI systems that can create new content (text, images, audio, code) based on patterns learned from training data.

**Generator**
In GANs, the network that creates synthetic data samples from random noise.

**Gradient**
The vector of partial derivatives of the loss function with respect to model parameters, indicating the direction of steepest increase.

**Gradient Descent**
An optimization algorithm that iteratively adjusts parameters in the direction that reduces the loss function.

**Gradient Clipping**
A technique to prevent exploding gradients by capping gradient values during training.

**GPT (Generative Pre-trained Transformer)**
A family of large language models developed by OpenAI that use decoder-only transformer architecture for text generation.

**GRU (Gated Recurrent Unit)**
A type of recurrent neural network similar to LSTM but with fewer parameters, using update and reset gates.

---

## H

**Hallucination**
When an AI model generates plausible but incorrect or fabricated information.

**Hyperparameter**
A parameter set before training begins (learning rate, batch size, etc.), not learned from data.
When AI models generate plausible-sounding but factually incorrect or nonsensical content.

**Hidden Layer**
Layers in a neural network between the input and output layers where intermediate computations occur.

**Hyperparameter**
Configuration settings external to the model that are set before training (e.g., learning rate, batch size, number of layers).

**Hyperparameter Tuning**
The process of finding optimal hyperparameter values, often using techniques like grid search, random search, or Bayesian optimization.

---

## I

**In-Context Learning**
The ability of LLMs to learn from examples provided in the prompt without parameter updates.
**Image Classification**
The task of assigning a label to an entire image from a predefined set of categories.

**Inference**
Using a trained model to make predictions on new data.

**Information Gain**
A measure of how much information a feature provides for classification. Used in decision trees.
**Information Retrieval**
The process of finding relevant information from a collection of documents, fundamental to search engines and RAG systems.

**Instance Segmentation**
Computer vision task that identifies and delineates individual objects in an image at the pixel level.

---

## J

**JAX**
Google's high-performance numerical computing library that combines NumPy with automatic differentiation and GPU/TPU acceleration.

---

## K

**Kernel**
1. In CNNs: A filter applied via convolution
2. In SVMs: A function that computes similarity in high-dimensional space

**K-Means**
A clustering algorithm that partitions data into K clusters based on distance to centroids.

**KL Divergence (Kullback-Leibler)**
A measure of how one probability distribution differs from another.
**K-Fold Cross-Validation**
A cross-validation method that divides data into K equal parts, using each part once as validation while training on the others.

**K-Means**
A clustering algorithm that partitions data into K clusters by iteratively assigning points to the nearest centroid.

**K-Nearest Neighbors (KNN)**
A simple algorithm that classifies new points based on the majority class of their K closest neighbors in feature space.

**Kernel**
In CNNs, a small matrix of weights that slides over the input to detect features. In SVMs, a function that computes similarity in a higher-dimensional space.

**Knowledge Distillation**
Training a smaller "student" model to mimic a larger "teacher" model, often achieving comparable performance with fewer parameters.

---

## L

**Label**
The target value or ground truth that a model learns to predict.

**Large Language Model (LLM)**
A neural network trained on massive text data, capable of understanding and generating language.

**Latent Space**
A lower-dimensional representation learned by models like autoencoders and VAEs.

**Learning Rate**
A hyperparameter controlling the step size during gradient descent.

**LSTM (Long Short-Term Memory)**
A type of RNN with gates that help capture long-term dependencies.

**Loss Function**
A function measuring the difference between predictions and actual values. Optimized during training.
The target variable or ground truth value associated with a training example in supervised learning.

**LangChain**
A framework for developing applications powered by language models, providing tools for chains, agents, and memory.

**Large Language Model (LLM)**
Neural networks with billions of parameters trained on massive text corpora, capable of understanding and generating human language.

**Latent Space**
The compressed representation space learned by models like autoencoders and VAEs, where similar items are mapped to nearby points.

**Layer Normalization**
A normalization technique that normalizes across features rather than batch dimension, commonly used in transformers.

**Learning Rate**
A hyperparameter controlling the step size during gradient descent optimization. Too high causes divergence; too low causes slow convergence.

**LIME (Local Interpretable Model-agnostic Explanations)**
A technique for explaining individual model predictions by approximating the model locally with an interpretable model.

**Linear Regression**
A statistical method for modeling the relationship between a dependent variable and one or more independent variables using a linear equation.

**LLaMA**
Meta's family of open-source large language models, designed to be efficient and accessible for research.

**Logistic Regression**
A classification algorithm that models the probability of a binary outcome using a logistic (sigmoid) function.

**LoRA (Low-Rank Adaptation)**
A parameter-efficient fine-tuning technique that adds small trainable matrices to frozen pre-trained weights.

**Loss Function**
A function that measures how well a model's predictions match the actual values. Training aims to minimize this function.

**LSTM (Long Short-Term Memory)**
A type of recurrent neural network with gates that control information flow, designed to capture long-range dependencies.

---

## M

**MLOps**
Practices for deploying, monitoring, and maintaining ML models in production.

**Multi-Head Attention**
Running multiple attention mechanisms in parallel, allowing the model to attend to different aspects simultaneously.

**Multimodal**
Models that process multiple types of data (text, images, audio).
**Machine Learning**
A subset of AI where systems learn from data to improve performance on tasks without being explicitly programmed.

**MAE (Mean Absolute Error)**
A regression metric that calculates the average absolute difference between predictions and actual values.

**Masked Language Modeling**
A pre-training task where the model predicts randomly masked tokens in a sequence, used by BERT and similar models.

**Max Pooling**
A downsampling operation that selects the maximum value from each region, commonly used in CNNs.

**Mean Pooling**
A downsampling operation that computes the average value from each region.

**Meta-Learning**
"Learning to learn" - algorithms that improve their learning ability through experience across multiple tasks.

**Mini-Batch**
A small subset of training examples used in one gradient update step.

**MLOps**
Practices for deploying and maintaining machine learning models in production, combining ML, DevOps, and data engineering.

**Model**
A mathematical representation learned from data that can make predictions or decisions.

**Momentum**
An optimization technique that accelerates gradient descent by accumulating a velocity vector in the direction of persistent gradients.

**MSE (Mean Squared Error)**
A regression metric that calculates the average squared difference between predictions and actual values.

**Multi-Head Attention**
Attention mechanism with multiple parallel attention operations, allowing the model to attend to different aspects simultaneously.

**Multi-Modal**
AI systems that can process and understand multiple types of data (text, images, audio) together.

---

## N

**Named Entity Recognition (NER)**
Identifying and classifying named entities (persons, locations, organizations) in text.

**Neural Network**
A computational model inspired by biological neurons, consisting of layers of interconnected nodes.

**NLP (Natural Language Processing)**
The field of AI dealing with language understanding and generation.

**Normalization**
Scaling features to a standard range. Examples: StandardScaler, MinMaxScaler.
NLP task of identifying and classifying named entities (people, organizations, locations) in text.

**Natural Language Processing (NLP)**
The field of AI focused on enabling computers to understand, interpret, and generate human language.

**Neural Network**
A computing system inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers.

**Normalization**
Techniques to scale data or layer outputs to a standard range or distribution, improving training stability.

---

## O

**One-Hot Encoding**
Representing categorical variables as binary vectors.

**Overfitting**
When a model learns training data too well, including noise, and fails to generalize.
**Object Detection**
Computer vision task of identifying and localizing objects within images, outputting bounding boxes and class labels.

**One-Hot Encoding**
A representation where categorical variables are converted to binary vectors with a single 1 and all other values 0.

**Optimizer**
An algorithm that updates model parameters to minimize the loss function. Examples include SGD, Adam, and RMSprop.

**Overfitting**
When a model learns the training data too well, including noise, resulting in poor generalization to new data.

---

## P

**Parameter**
A value learned during training (weights and biases in neural networks).

**Perceptron**
The simplest neural network unit, computing a weighted sum of inputs.

**Perplexity**
A metric for language models. Lower perplexity indicates better prediction of text.

**Pooling**
Downsampling operation in CNNs (max pooling, average pooling).

**Precision**
The proportion of positive predictions that are correct. TP / (TP + FP).

**Pre-training**
Training a model on a large dataset before fine-tuning on a specific task.

**Prompt Engineering**
Designing effective prompts to get desired outputs from LLMs.
Learnable weights in a model that are adjusted during training to minimize the loss function.

**Perceptron**
The simplest form of neural network - a single neuron with weighted inputs, bias, and an activation function.

**Perplexity**
A metric for evaluating language models that measures how well the model predicts a sample. Lower is better.

**Pooling**
An operation that reduces spatial dimensions by aggregating values within regions (e.g., max pooling, average pooling).

**PPO (Proximal Policy Optimization)**
A popular reinforcement learning algorithm that uses a clipped objective to ensure stable policy updates.

**Pre-training**
Training a model on a large dataset to learn general representations before fine-tuning on a specific task.

**Precision**
The proportion of positive predictions that are actually correct: TP / (TP + FP).

**Prompt**
The input text given to a language model to guide its output.

**Prompt Engineering**
The practice of crafting effective prompts to elicit desired behaviors from language models.

---

## Q

**Q-Learning**
A reinforcement learning algorithm that learns the value of taking each action in each state without requiring a model of the environment.

**Quantization**
Reducing the precision of model weights (e.g., from 32-bit to 8-bit) to reduce memory usage and speed up inference.

---

## R

**RAG (Retrieval-Augmented Generation)**
Combining retrieval from a knowledge base with LLM generation for more accurate responses.

**Recall**
The proportion of actual positives correctly identified. TP / (TP + FN).

**Regression**
Predicting a continuous numerical value.

**Regularization**
Techniques to prevent overfitting (L1, L2, dropout).

**Reinforcement Learning (RL)**
Learning through interaction with an environment to maximize reward.

**Reinforcement Learning from Human Feedback (RLHF)**
Training models using human preferences as reward signals.

**ReLU (Rectified Linear Unit)**
An activation function: f(x) = max(0, x).

**ResNet (Residual Network)**
A CNN architecture using skip connections to enable very deep networks.

**RNN (Recurrent Neural Network)**
A neural network that processes sequential data by maintaining hidden state.
**R² Score (Coefficient of Determination)**
A regression metric indicating the proportion of variance in the target explained by the model. Ranges from -∞ to 1.

**RAG (Retrieval-Augmented Generation)**
A technique that enhances language models by retrieving relevant information from external sources before generating responses.

**Random Forest**
An ensemble method that creates multiple decision trees on random subsets of data and features, then aggregates their predictions.

**Recall**
The proportion of actual positives that were correctly identified: TP / (TP + FN). Also called sensitivity.

**Regularization**
Techniques to prevent overfitting by adding constraints or penalties to the model (e.g., L1, L2 regularization, dropout).

**Reinforcement Learning (RL)**
A learning paradigm where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards.

**ReLU (Rectified Linear Unit)**
An activation function that returns max(0, x), introducing non-linearity while being computationally efficient.

**Representation Learning**
Automatically discovering useful representations or features from raw data.

**Residual Connection (Skip Connection)**
A shortcut connection that adds the input of a layer to its output, enabling training of very deep networks.

**RLHF (Reinforcement Learning from Human Feedback)**
A technique for fine-tuning language models using human preferences to guide learning.

**RMSE (Root Mean Squared Error)**
The square root of MSE, providing error in the same units as the target variable.

**RNN (Recurrent Neural Network)**
A neural network architecture for sequential data where outputs from previous steps are fed back as inputs.

**ROC Curve (Receiver Operating Characteristic)**
A plot of true positive rate vs. false positive rate at various classification thresholds.

---

## S

**Self-Attention**
Attention mechanism where a sequence attends to itself. Core component of Transformers.

**Softmax**
A function that converts a vector of scores into a probability distribution.

**Stochastic Gradient Descent (SGD)**
Gradient descent using random subsets (mini-batches) of data.

**Supervised Learning**
Learning from labeled data (input-output pairs).

**SVM (Support Vector Machine)**
A classification algorithm that finds the optimal hyperplane separating classes.
**Sampling (Generative)**
Techniques for generating diverse outputs from probabilistic models (e.g., temperature, top-k, top-p sampling).

**Segmentation**
Computer vision task of classifying each pixel in an image into categories.

**Self-Attention**
An attention mechanism where each position in a sequence attends to all positions, enabling modeling of dependencies regardless of distance.

**Self-Supervised Learning**
Learning representations from unlabeled data by creating supervised signals from the data itself (e.g., predicting masked tokens).

**Semantic Segmentation**
Classifying each pixel in an image into a category without distinguishing between instances of the same class.

**Sentiment Analysis**
NLP task of determining the emotional tone or opinion expressed in text.

**Sequence-to-Sequence (Seq2Seq)**
A model architecture for transforming one sequence into another, used in machine translation and summarization.

**SGD (Stochastic Gradient Descent)**
Gradient descent using random samples (mini-batches) rather than the full dataset for each update.

**SHAP (SHapley Additive exPlanations)**
An explainability method based on game theory that assigns importance values to each feature for individual predictions.

**Sigmoid**
An activation function that squashes values to the range (0, 1): σ(x) = 1 / (1 + e^(-x)).

**Softmax**
A function that converts a vector of numbers into a probability distribution that sums to 1.

**Supervised Learning**
Learning from labeled examples where both inputs and desired outputs are provided during training.

**SVM (Support Vector Machine)**
A classification algorithm that finds the optimal hyperplane separating different classes with maximum margin.

---

## T

**Temperature**
A parameter controlling randomness in LLM outputs. Higher = more random.

**TensorFlow**
An open-source deep learning framework by Google.

**Token**
A unit of text (word, subword, or character) processed by language models.

**Transfer Learning**
Using knowledge from one task to improve performance on another.

**Transformer**
An architecture based entirely on attention mechanisms. Foundation of modern NLP.
A parameter controlling randomness in model output generation. Higher values produce more diverse but less focused outputs.

**TensorFlow**
Google's open-source machine learning framework for building and deploying ML models.

**Tokenization**
The process of breaking text into smaller units (tokens) that can be processed by NLP models.

**Top-k Sampling**
A text generation technique that samples from only the k most probable next tokens.

**Top-p (Nucleus) Sampling**
A text generation technique that samples from the smallest set of tokens whose cumulative probability exceeds p.

**Transfer Learning**
Using knowledge learned from one task to improve performance on a different but related task.

**Transformer**
A neural network architecture based entirely on attention mechanisms, introduced in "Attention Is All You Need." Foundation of modern NLP.

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
A dimensionality reduction technique for visualizing high-dimensional data in 2D or 3D.

---

## U

**Underfitting**
When a model is too simple to capture patterns in the data.

**Unsupervised Learning**
Learning patterns from unlabeled data (clustering, dimensionality reduction).
**UMAP (Uniform Manifold Approximation and Projection)**
A dimensionality reduction technique similar to t-SNE but often faster and better at preserving global structure.

**Underfitting**
When a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data.

**Unsupervised Learning**
Learning patterns from data without labeled examples, including clustering and dimensionality reduction.

---

## V

**Validation Set**
A subset of data used to tune hyperparameters and evaluate during training.

**VAE (Variational Autoencoder)**
A generative model that learns a probabilistic latent space.

**Vanishing Gradient**
When gradients become too small during backpropagation, preventing learning in early layers.

**Variance**
In bias-variance tradeoff: error from sensitivity to small fluctuations in training data.

**Vector Database**
A database optimized for storing and searching high-dimensional vectors (embeddings).

**Vision Transformer (ViT)**
Applying the Transformer architecture to image classification.
A subset of data used to tune hyperparameters and make decisions during model development, separate from training and test sets.

**VAE (Variational Autoencoder)**
A generative model that learns a probabilistic mapping between data and a latent space, enabling generation of new samples.

**Vanishing Gradient**
A problem where gradients become extremely small during backpropagation in deep networks, preventing effective learning.

**Variance**
In the bias-variance tradeoff, the amount by which model predictions change with different training data. High variance leads to overfitting.

**Vector Database**
A database optimized for storing and searching high-dimensional vectors, essential for similarity search and RAG applications.

**Vision Transformer (ViT)**
A transformer architecture adapted for image classification by treating images as sequences of patches.

---

## W

**Weight**
Learnable parameters in a neural network that are multiplied with inputs.

**Word2Vec**
An algorithm for learning word embeddings from text.
**Weights**
The learnable parameters in a neural network that determine how inputs are transformed.

**Word2Vec**
A technique for learning word embeddings by predicting context words (Skip-gram) or predicting a word from context (CBOW).

**Word Embedding**
Dense vector representations of words where semantically similar words have similar vectors.

---

## X

**XGBoost**
An optimized gradient boosting library known for high performance on tabular data.
**XAI (Explainable AI)**
The field focused on making AI systems' decisions interpretable and understandable to humans.

**XGBoost**
An optimized gradient boosting library known for speed and performance in tabular data competitions.

---

## Y

**YOLO (You Only Look Once)**
A family of real-time object detection models that process images in a single pass through the network.

---

## Z

**Zero-Shot Learning**
Making predictions on classes not seen during training. In LLMs, completing tasks without examples.

---

🌐 [Back to Main Repository](../README.md) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
The ability of a model to perform tasks it wasn't explicitly trained on, often through clever prompting or transfer learning.

**Zero-Shot Classification**
Classifying examples into categories the model has never seen during training, typically using semantic understanding.

---

<div align="center">

## 📚 Continue Learning

| Section | Link |
|---------|------|
| 📚 Notes | [Browse Notes →](../notes/README.md) |
| 💻 Code Examples | [Browse Code →](../code/README.md) |
| 🔗 Resources | [Browse Resources →](../resources/README.md) |
| 📋 Cheatsheets | [Browse Cheatsheets →](../cheatsheets/README.md) |

---

[← Back to Main](../README.md)

</div>
