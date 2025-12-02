# 📚 AI/ML Glossary

A comprehensive dictionary of AI and Machine Learning terminology.

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

---

## H

**Hallucination**
When an AI model generates plausible but incorrect or fabricated information.

**Hyperparameter**
A parameter set before training begins (learning rate, batch size, etc.), not learned from data.

---

## I

**In-Context Learning**
The ability of LLMs to learn from examples provided in the prompt without parameter updates.

**Inference**
Using a trained model to make predictions on new data.

**Information Gain**
A measure of how much information a feature provides for classification. Used in decision trees.

---

## K

**Kernel**
1. In CNNs: A filter applied via convolution
2. In SVMs: A function that computes similarity in high-dimensional space

**K-Means**
A clustering algorithm that partitions data into K clusters based on distance to centroids.

**KL Divergence (Kullback-Leibler)**
A measure of how one probability distribution differs from another.

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

---

## M

**MLOps**
Practices for deploying, monitoring, and maintaining ML models in production.

**Multi-Head Attention**
Running multiple attention mechanisms in parallel, allowing the model to attend to different aspects simultaneously.

**Multimodal**
Models that process multiple types of data (text, images, audio).

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

---

## O

**One-Hot Encoding**
Representing categorical variables as binary vectors.

**Overfitting**
When a model learns training data too well, including noise, and fails to generalize.

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

---

## U

**Underfitting**
When a model is too simple to capture patterns in the data.

**Unsupervised Learning**
Learning patterns from unlabeled data (clustering, dimensionality reduction).

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

---

## W

**Weight**
Learnable parameters in a neural network that are multiplied with inputs.

**Word2Vec**
An algorithm for learning word embeddings from text.

---

## X

**XGBoost**
An optimized gradient boosting library known for high performance on tabular data.

---

## Z

**Zero-Shot Learning**
Making predictions on classes not seen during training. In LLMs, completing tasks without examples.

---

🌐 [Back to Main Repository](../README.md) | 🔗 [Visit jgcks.com](https://www.jgcks.com)
