# WEEK 8: CONVOLUTIONAL NEURAL NETWORKS

## 1. From Fully Connected to Convolutional

### 1.1 The Need for CNNs

Traditional fully connected neural networks face significant challenges when dealing with image data. Consider an input image of size 224x224x3 (RGB). In a fully connected network, this would require 150,528 input neurons, and if the first hidden layer had 1000 neurons, we would need over 150 million parameters for just this first connection. This approach is both computationally inefficient and prone to overfitting.

Convolutional Neural Networks (CNNs) address these challenges through three key insights:
1. Local connectivity: Neurons only connect to a small region of the input
2. Parameter sharing: The same filters are applied across the entire image
3. Translation invariance: Features can be detected regardless of their position

### 1.2 Basic CNN Operations

At its core, convolution is a mathematical operation that combines two functions to produce a third function. In the context of CNNs, think of it as sliding a small window (the kernel or filter) across an image and performing a weighted sum at each position. This process is analogous to using a spotlight to systematically examine parts of an image - the spotlight (kernel) highlights specific features like edges, textures, or patterns. When a pattern in the image matches the kernel's pattern, it produces a strong activation in the output feature map. This simple yet powerful operation allows CNNs to automatically learn and detect important visual features.

The fundamental operation in CNNs is convolution, which can be implemented as:

```python
import numpy as np

def convolution2d(image, kernel):
    # Assuming 'valid' padding
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    output_height = i_height - k_height + 1
    output_width = i_width - k_width + 1
    
    output = np.zeros((output_height, output_width))
    
    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(
                image[y:y+k_height, x:x+k_width] * kernel
            )
    
    return output
```

This operation creates feature maps that respond to specific patterns in the input image. The kernel (or filter) acts as a pattern detector, sliding across the image to identify where these patterns occur.

## 2. Core CNN Components

### 2.1 Convolutional Layers

The convolutional layer is defined by several key parameters:
- Kernel size: The spatial extent of the filter (e.g., 3x3, 5x5)
- Stride: The step size when sliding the filter
- Padding: How to handle the image borders
- Number of filters: How many different patterns to detect

```python
import torch.nn as nn

class BasicConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same'
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

### 2.2 Pooling Operations

Pooling layers serve to:
1. Reduce spatial dimensions
2. Build in translation invariance
3. Control overfitting

The two main types are:

```python
# Max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

### 2.3 Activation Functions

ReLU (Rectified Linear Unit) has become the standard activation function in CNNs due to:
- Fast computation
- No vanishing gradient for positive values
- Sparse activation

## 3. ResNet Architecture

### 3.1 The Vanishing Gradient Problem

As networks become deeper, the gradient signal becomes weaker as it propagates backward through the layers. ResNet addresses this through skip connections:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = torch.relu(out)
        return out
```

## 4. Transfer Learning with CNNs

### 4.1 Pre-trained Models

Modern CNN development rarely starts from scratch. Instead, we leverage pre-trained models:

```python
def create_transfer_model(num_classes, freeze_backbone=True):
    model = models.resnet50(pretrained=True)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

### 4.2 Fine-tuning Strategies

The choice of fine-tuning strategy depends on:
1. Size of target dataset
2. Similarity to source domain
3. Available computational resources

```python
def staged_training(model, epochs_per_stage):
    # Stage 1: Train only new layers
    for param in model.parameters():
        param.requires_grad = False
    model.fc.requires_grad = True
    train_model(model, epochs_per_stage)
    
    # Stage 2: Fine-tune all layers
    for param in model.parameters():
        param.requires_grad = True
    train_model(model, epochs_per_stage)
```

### 4.3 Domain Adaptation

When transferring between domains, we need to handle domain shift:

```python
class DomainAdaptationModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.feature_extractor = backbone
        self.classifier = nn.Linear(2048, num_classes)
        self.domain_classifier = nn.Linear(2048, 2)
    
    def forward(self, x, alpha):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        
        # Gradient reversal for domain adaptation
        reverse_features = GradientReversal.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        
        return class_output, domain_output
```

## Summary
- CNNs provide efficient processing of image data
- Modern architectures build on proven components
- Transfer learning enables rapid application
- Domain adaptation extends model applicability 