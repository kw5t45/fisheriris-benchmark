# Fisher Iris Benchmark – MLP vs ELM

This repository contains Python code for comparing **Multi-Layer Perceptron (MLP)** and **Extreme Learning Machine (ELM)** models on the **Fisher Iris dataset**. The goal is to evaluate model performance with different architectures and hidden neuron configurations.

## Dataset

- **Fisher Iris** dataset
- 150 samples, 3 classes (`Iris setosa`, `Iris versicolor`, `Iris virginica`)
- 4 features per sample (`sepal length`, `sepal width`, `petal length`, `petal width`)
- Fully labeled and clean, ideal for classification experiments

## Model Parameters

### MLP
- Hidden layers: 1–5 layers, 1–5 neurons per layer
- Activation function: Sigmoid (hidden layers), Softmax (output)
- Learning rate: 0.01
- Optimizer: Stochastic Gradient Descent
- Loss function: Mean Squared Error (MSE)
- Epochs: 500
- Validation: K-Fold Cross Validation (5 folds)
- Encoding: One-hot for output classes

### ELM
- Hidden neurons: 1–25
- Activation function: Sigmoid (hidden layer), Softmax (output)
- Training: Closed-form solution for output weights
- Validation: K-Fold Cross Validation (5 folds)

## Results

| MLP Architecture | MLP Training Error | MLP Validation Error | ELM Hidden Neurons | ELM Validation Error |
|-----------------|-----------------|-------------------|-----------------|-------------------|
| 1               | 0.224           | 0.224             | 1               | **0.206**         |
| 2x1             | 0.224           | 0.224             | 2               | **0.193**         |
| 3x1             | 0.227           | 0.228             | 3               | **0.158**         |
| 4x1             | 0.230           | 0.232             | 4               | **0.148**         |
| 5x1             | 0.224           | 0.224             | 5               | **0.146**         |
| 2               | 0.216           | 0.216             | 6               | **0.137**         |
| 2x2             | 0.223           | 0.224             | 7               | **0.120**         |
| 3x2             | 0.225           | 0.226             | 8               | **0.127**         |
| 4x2             | 0.228           | 0.231             | 9               | **0.118**         |
| 5x2             | 0.225           | 0.224             | 10              | **0.124**         |
| 3               | 0.208           | 0.209             | 11              | **0.114**         |
| 2x3             | 0.226           | 0.226             | 12              | **0.117**         |
| 3x3             | 0.223           | 0.225             | 13              | **0.119**         |
| 4x3             | 0.224           | 0.227             | 14              | **0.106**         |
| 5x3             | 0.223           | 0.224             | 15              | **0.111**         |
| 4               | 0.204           | 0.208             | 16              | **0.111**         |
| 2x4             | 0.222           | 0.223             | 17              | **0.107**         |
| 3x4             | 0.222           | 0.224             | 18              | **0.108**         |
| 4x4             | 0.224           | 0.224             | 19              | **0.105**         |
| 5x4             | 0.222           | 0.223             | 20              | **0.103**         |
| 5               | 0.201           | 0.204             | 21              | **0.105**         |
| 2x5             | 0.224           | 0.225             | 22              | **0.104**         |
| 3x5             | 0.222           | 0.223             | 23              | **0.107**         |
| 4x5             | 0.225           | 0.229             | 24              | **0.103**         |
| 5x5             | 0.223           | 0.223             | 25              | **0.104**         |

## Key Observations
- MLP shows relatively stable validation error across different architectures.
- ELM shows a consistent decrease in error as hidden neurons increase.
- ELM is significantly faster to train (~0.15s) compared to MLP (~42.85s).

## Code
The code is written in Python 3.11 and can be found [here](https://github.com/kw5t45/fisheriris-benchmark).

