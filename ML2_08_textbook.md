# WEEK 8: CONVOLUTIONAL NEURAL NETWORKS

## From Fully Connected to Convolutional

The transition from fully connected neural networks to convolutional neural networks (CNNs) is motivated by several key factors that make CNNs particularly well-suited for processing grid-like data, such as images. One of the primary motivations is parameter efficiency. In a fully connected network, each neuron is connected to every neuron in the previous layer, resulting in a large number of parameters that need to be learned. This can lead to overfitting and computational inefficiency, especially when dealing with high-dimensional data like images. CNNs address this issue by utilizing local connectivity and weight sharing, which significantly reduces the number of parameters while still capturing the essential features of the input.

Another important motivation for CNNs is their ability to capture spatial relationships within the input data. In an image, nearby pixels are often highly correlated and contain meaningful information about the local structure and patterns. CNNs exploit this spatial structure by applying convolutional filters that operate on local regions of the input, allowing the network to learn and detect local features such as edges, corners, and textures. By stacking multiple convolutional layers, CNNs can hierarchically learn more complex and abstract features, enabling them to capture both low-level and high-level patterns in the data.

Translation invariance is another key property that makes CNNs well-suited for image processing tasks. In many cases, the presence of a particular feature or object in an image is more important than its exact location. CNNs achieve translation invariance through the use of pooling operations, which downsample the feature maps and provide a degree of spatial invariance. This allows the network to recognize objects or patterns regardless of their precise position in the input, making CNNs robust to small translations and distortions.

The basic building blocks of CNNs are convolutional filters, feature maps, and receptive fields. Convolutional filters are small matrices of learnable weights that are convolved with the input to produce feature maps. Each filter is designed to detect a specific local pattern or feature, and the resulting feature map represents the presence or absence of that feature at different locations in the input. The size of the convolutional filter determines the receptive field, which is the region of the input that influences a particular neuron in the feature map. By applying multiple filters and stacking convolutional layers, CNNs can learn a rich set of features at different scales and abstractions.

Mathematically, the convolution operation can be expressed as:

$$y(i,j) = \sum_{m}\sum_{n} x(i+m, j+n) \cdot w(m,n)$$

where $y(i,j)$ represents the output feature map at position $(i,j)$, $x(i+m, j+n)$ represents the input patch centered at $(i+m, j+n)$, and $w(m,n)$ represents the convolutional filter weights. The convolution operation is performed by sliding the filter over the input, computing the element-wise product between the filter and the input patch, and summing the results to obtain the output value at each position. This operation is repeated for multiple filters to generate a set of feature maps that capture different aspects of the input.


## Core CNN Components

Convolutional Neural Networks (CNNs) are a class of deep learning models specifically designed for processing grid-like data, such as images. The core components of CNNs are convolutional layers, pooling operations, and activation functions. Convolutional layers are the fundamental building blocks of CNNs, responsible for extracting features from the input data. These layers consist of a set of learnable filters, also known as kernels, which are convolved with the input to produce feature maps. The size of the kernel, denoted as the kernel size, determines the receptive field of each neuron in the convolutional layer. The stride, which is the step size at which the kernel moves across the input, controls the spatial resolution of the output feature map. Padding options, such as zero-padding or valid padding, are used to control the size of the output feature map and preserve spatial information at the borders. Additionally, convolutional layers can have multiple channels, allowing for the extraction of different types of features from the input.

Pooling operations are another crucial component of CNNs, used to reduce the spatial dimensions of the feature maps while retaining the most important information. The two most common types of pooling are max pooling and average pooling. Max pooling selects the maximum value within a specified window, while average pooling computes the average value within the window. By applying pooling operations, CNNs can achieve spatial reduction, which helps to reduce the computational complexity and provides a form of translation invariance. The size of the pooling window and the stride determine the extent of spatial reduction. Pooling operations are typically applied after one or more convolutional layers, allowing the network to progressively focus on higher-level features.

Activation functions play a vital role in CNNs by introducing non-linearity into the network. The most commonly used activation function in CNNs is the Rectified Linear Unit (ReLU), defined as $$f(x) = max(0, x)$$. ReLU activation is applied element-wise to the output of convolutional layers, effectively removing negative values and creating a sparse representation. This non-linearity is crucial for CNNs to learn complex patterns and representations from the input data. The application of ReLU activation to the feature maps allows the network to capture non-linear relationships and increase the expressive power of the model.

The combination of convolutional layers, pooling operations, and activation functions enables CNNs to learn hierarchical representations of the input data. As the network deepens, the convolutional layers capture increasingly abstract and high-level features. The spatial reduction achieved through pooling operations helps to make the network more robust to small translations and distortions in the input. The non-linearity introduced by activation functions allows the network to learn complex decision boundaries and capture intricate patterns in the data.

The design of CNN architectures involves carefully selecting the number and size of convolutional layers, the type and size of pooling operations, and the choice of activation functions. The specific configuration of these components depends on the nature of the task and the complexity of the input data. For example, deeper networks with more convolutional layers are often used for more challenging tasks, such as object detection or semantic segmentation. The size of the kernels and the stride of the convolutional layers can be adjusted to capture features at different scales and resolutions. The choice of pooling operations and their placement in the network can impact the trade-off between spatial resolution and feature abstraction. Activation functions, such as ReLU, are crucial for introducing non-linearity and enabling the network to learn complex representations. By carefully designing and tuning these core components, CNNs can achieve state-of-the-art performance on a wide range of computer vision tasks.

## ResNet Architecture

ResNet, short for Residual Networks, is a groundbreaking architecture in deep learning that has revolutionized the field of computer vision. The core idea behind ResNet is the introduction of residual learning, which addresses the problem of vanishing gradients in deep neural networks. By employing skip connections, also known as identity mappings, ResNet allows the gradients to flow directly through the network, enabling the training of much deeper models without suffering from performance degradation.

The skip connections in ResNet work by adding the input of a block to its output, effectively creating a shortcut that bypasses one or more layers. This design choice allows the network to learn residual functions, which are the differences between the desired mapping and the identity mapping. By learning these residual functions, ResNet can capture more complex patterns and representations in the data. Moreover, the identity mappings ensure that the gradients can propagate smoothly through the network, mitigating the vanishing gradient problem that plagues deep neural networks.

ResNet comes in various configurations, with the most commonly used variants being ResNet-50, ResNet-101, and ResNet-152. These numbers indicate the depth of the network, i.e., the number of layers. ResNet-50 consists of 50 layers, ResNet-101 has 101 layers, and ResNet-152 comprises 152 layers. The increased depth of these models allows them to learn more intricate features and representations from the input data. However, the deeper the network, the more computationally expensive it becomes, both in terms of training time and memory requirements.

The impact of ResNet on the field of deep learning cannot be overstated. By solving the vanishing gradient problem, ResNet has enabled the training of extremely deep neural networks, surpassing the performance of previous architectures. This breakthrough has led to significant advancements in various computer vision tasks, such as image classification, object detection, and semantic segmentation. ResNet's ability to learn hierarchical features at different scales has proven to be highly effective in capturing both low-level and high-level patterns in images.

The mathematical formulation of a residual block in ResNet can be expressed as follows:

$$x_{l+1} = x_l + \mathcal{F}(x_l, \mathcal{W}_l)$$

where $x_l$ represents the input to the $l$-th residual block, $\mathcal{F}$ denotes the residual function, $\mathcal{W}_l$ represents the weights of the $l$-th block, and $x_{l+1}$ is the output of the block. The residual function $\mathcal{F}$ typically consists of multiple convolutional layers, batch normalization, and activation functions. By adding the input $x_l$ to the output of the residual function, the network can learn the identity mapping if necessary, allowing for a smooth flow of information and gradients through the network.

## Transfer Learning with CNNs

Transfer learning is a powerful technique in deep learning that allows leveraging pre-trained models to solve new tasks with limited labeled data. In the context of Convolutional Neural Networks (CNNs), transfer learning is particularly effective due to the hierarchical nature of learned features. Pre-trained models, such as those trained on the ImageNet dataset, have learned a rich set of features that can be transferred to new domains. These models act as feature extractors, where the lower layers capture general features like edges and textures, while the higher layers capture more task-specific features. By freezing certain layers of the pre-trained model and fine-tuning others, we can adapt the model to new tasks efficiently.

Fine-tuning approaches in transfer learning can be categorized into three main strategies: full fine-tuning, partial fine-tuning, and linear probing. Full fine-tuning involves updating all the layers of the pre-trained model during training on the new task. This allows the model to adapt its features entirely to the new domain but requires more computational resources and may lead to overfitting if the new dataset is small. Partial fine-tuning, on the other hand, freezes some of the lower layers and fine-tunes only the higher layers. This approach retains the general features learned by the pre-trained model while allowing the higher layers to specialize for the new task. Linear probing is a more restrictive approach where all the layers of the pre-trained model are frozen, and only a new linear classifier is trained on top of the extracted features. This is useful when the new dataset is very small, and fine-tuning the entire model may lead to overfitting.

Domain adaptation is a crucial aspect of transfer learning, especially when the source domain (where the pre-trained model was trained) and the target domain (where the model is applied) have different data distributions. Cross-domain transfer learning aims to bridge this domain gap by learning domain-invariant features or by explicitly aligning the feature distributions of the source and target domains. Techniques such as adversarial domain adaptation and domain-specific batch normalization have been proposed to address this challenge. Few-shot learning is another scenario where transfer learning shines, as it enables learning from very few labeled examples in the target domain by leveraging the knowledge acquired from the source domain.

Handling domain shift is a critical consideration in transfer learning. Domain shift refers to the differences in data distributions between the source and target domains, which can lead to performance degradation if not addressed properly. Various techniques have been proposed to mitigate domain shift, such as domain adversarial training, where a discriminator network is trained to distinguish between source and target domains, and the feature extractor is trained to confuse the discriminator. Another approach is to use domain-specific normalization techniques, such as adaptive batch normalization, which adjusts the batch normalization statistics separately for each domain. Additionally, data augmentation techniques specific to the target domain can help in reducing the impact of domain shift.

The effectiveness of transfer learning with CNNs has been demonstrated in numerous applications, ranging from medical image analysis to remote sensing. For example, in medical image classification tasks, pre-trained models like ResNet and DenseNet have been successfully fine-tuned to detect diseases from X-ray images, achieving state-of-the-art performance with limited labeled data. Similarly, in remote sensing, pre-trained models have been adapted for tasks such as land cover classification and object detection, significantly reducing the need for large annotated datasets. Transfer learning has also been applied to video analysis tasks, such as action recognition and video captioning, by leveraging pre-trained models from the image domain and extending them to handle temporal information. As the field of deep learning continues to evolve, transfer learning with CNNs remains a fundamental technique for efficiently solving a wide range of computer vision problems.

# Old
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