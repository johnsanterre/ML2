# Neural Networks Fundamentals & Backpropagation

## Introduction
Understanding backpropagation and the fundamentals of neural network training is crucial for deep learning. This chapter explores the mathematical foundations and practical implementations of gradient-based learning in neural networks.

## 1. Neural Network Training Fundamentals

### 1.1 Forward Propagation
The forward pass computes predictions by propagating input through the network:

```python
def forward(x, weights, biases):
    # First layer
    z1 = np.dot(x, weights[0]) + biases[0]
    a1 = relu(z1)
    
    # Second layer
    z2 = np.dot(a1, weights[1]) + biases[1]
    a2 = relu(z2)
    
    # Output layer
    z3 = np.dot(a2, weights[2]) + biases[2]
    y_pred = z3  # Linear output for regression
    
    return y_pred, (z1, a1, z2, a2, z3)
```

### 1.2 Computational Graphs
Neural networks can be represented as directed acyclic graphs:
- Nodes represent operations
- Edges represent data flow
- Gradients flow backward through the graph

Example for a simple two-layer network:
```
x → [Linear] → [ReLU] → [Linear] → [ReLU] → [Linear] → y
```

## 2. Understanding Backpropagation

### 2.1 The Chain Rule
Backpropagation applies the chain rule of calculus:

For a composite function f(g(x)):
∂f/∂x = ∂f/∂g × ∂g/∂x

In neural networks:
```python
def backward(grad_output, cached_values, weights):
    x, z1, a1, z2, a2 = cached_values
    
    # Gradient for last layer
    grad_w3 = np.dot(a2.T, grad_output)
    grad_b3 = np.sum(grad_output, axis=0)
    
    # Gradient for second layer
    grad_a2 = np.dot(grad_output, weights[2].T)
    grad_z2 = grad_a2 * relu_derivative(z2)
    
    # Continue backward through network...
```

### 2.2 Gradient Computation
Each layer computes local gradients:

1. Linear Layer:
   ```python
   def linear_backward(grad_output, input_data, weights):
       grad_input = np.dot(grad_output, weights.T)
       grad_weights = np.dot(input_data.T, grad_output)
       grad_bias = np.sum(grad_output, axis=0)
       return grad_input, grad_weights, grad_bias
   ```

2. ReLU Layer:
   ```python
   def relu_backward(grad_output, input_data):
       return grad_output * (input_data > 0)
   ```

## 3. Loss Functions

### 3.1 Mean Squared Error (MSE)
Common for regression problems:

```python
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_gradient(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_pred.shape[0]
```

### 3.2 Cross-Entropy Loss
Standard for classification:

```python
def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]

def cross_entropy_gradient(y_pred, y_true):
    return (y_pred - y_true) / y_pred.shape[0]
```

## 4. Gradient-Based Optimization

### 4.1 Stochastic Gradient Descent (SGD)
Basic optimization algorithm:

```python
def sgd_update(params, grads, learning_rate):
    for param, grad in zip(params, grads):
        param -= learning_rate * grad
```

### 4.2 Common Challenges

#### Vanishing Gradients
- Problem: Gradients become very small in early layers
- Solutions:
  * ReLU activation functions
  * Proper initialization
  * Residual connections

```python
def initialize_weights(layer_dims):
    weights = []
    for i in range(len(layer_dims)-1):
        # He initialization for ReLU
        w = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2/layer_dims[i])
        weights.append(w)
    return weights
```

#### Exploding Gradients
- Problem: Gradients become very large
- Solutions:
  * Gradient clipping
  * Layer normalization

```python
def clip_gradients(gradients, max_norm):
    total_norm = np.sqrt(sum(np.sum(grad ** 2) for grad in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in gradients:
            grad *= clip_coef
```

## 5. Training Loop Implementation

### 5.1 Basic Training Loop
```python
def train(model, dataloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            # Forward pass
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
```

### 5.2 Learning Rate Scheduling
Adapting the learning rate during training:

```python
def cosine_scheduler(initial_lr, epochs):
    def schedule(epoch):
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
    return schedule
```

## Summary
Understanding backpropagation and neural network training involves:
1. Forward propagation mechanics
2. Gradient computation through chain rule
3. Loss function selection and implementation
4. Optimization techniques and challenges
5. Practical training considerations

These fundamentals form the basis for more advanced deep learning concepts and architectures. 