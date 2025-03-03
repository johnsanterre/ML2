# Introduction to Deep Learning

## 1. Evolution of Machine Learning

### 1.1 Historical Context
The progression from traditional ML to deep learning:
- Rule-based systems (1950s-1960s)
- Classical ML algorithms (1970s-1990s)
- Deep learning revolution (2010s-present)

Key breakthroughs:
```python
# Example: Traditional ML vs Deep Learning approach
# Traditional ML: Manual feature engineering
def traditional_features(image):
    edges = detect_edges(image)
    corners = detect_corners(image)
    textures = compute_textures(image)
    return np.concatenate([edges, corners, textures])

# Deep Learning: Learned feature extraction
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
```

### 1.2 Why Deep Learning?
Advantages over traditional ML:
- Automatic feature learning
- Better scaling with data
- Hierarchical representations
- End-to-end learning

## 2. Neural Network Fundamentals

### 2.1 Basic Building Blocks
The neuron as a computational unit:

```python
class Neuron:
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()
    
    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)
    
    def activation(self, z):
        return max(0, z)  # ReLU activation
```

### 2.2 Layer Types
Common neural network layers:

```python
import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        # Initialize with He initialization
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2/input_dim)
        self.bias = np.zeros(output_dim)
    
    def forward(self, x):
        return np.dot(x, self.weights) + self.bias
```

## 3. Deep Learning Frameworks

### 3.1 PyTorch
Modern dynamic computation framework:

```python
import torch
import torch.nn as nn

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Training loop
def train_step(model, data, target, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()
```

### 3.2 TensorFlow/Keras
Static computation graph approach:

```python
import tensorflow as tf

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile and train
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
```

## 4. Basic Training Concepts

### 4.1 Data Handling
Efficient data loading and preprocessing:

```python
def prepare_data(X, y, batch_size=32):
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X),
        torch.LongTensor(y)
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
```

### 4.2 Training Loop
Basic training structure:

```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
```

## 5. Deep Learning Applications

### 5.1 Computer Vision
Basic image processing:

```python
class ConvolutionalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### 5.2 Natural Language Processing
Text processing basics:

```python
class TextNet(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])
```

## 6. Best Practices

### 6.1 Model Development
Key principles:
- Start simple
- Validate frequently
- Monitor metrics
- Debug systematically

```python
def develop_model():
    # 1. Start with baseline
    model = SimpleNet()
    
    # 2. Add validation
    val_score = validate(model, val_loader)
    
    # 3. Monitor metrics
    logger.log_metrics({
        'val_score': val_score,
        'model_size': count_parameters(model)
    })
    
    # 4. Systematic improvements
    model = add_improvements(model)
    return model
```

### 6.2 Common Pitfalls
Avoiding typical mistakes:
- Not normalizing inputs
- Wrong learning rate
- Poor initialization
- Inadequate validation

## Summary
Key takeaways:
1. Deep learning automates feature engineering
2. Framework choice affects development workflow
3. Good practices are essential for success
4. Start simple and iterate 