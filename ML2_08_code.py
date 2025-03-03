"""
Week 8: Convolutional Neural Networks
Code Examples and Implementations

This module demonstrates:
1. Basic CNN operations
2. Building CNN architectures
3. ResNet implementation
4. Transfer learning examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from typing import Tuple, List

class BasicConvolutions:
    """Demonstrates fundamental CNN operations."""
    
    @staticmethod
    def create_edge_detector() -> np.ndarray:
        """Create a simple horizontal edge detection kernel."""
        return np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ])
    
    @staticmethod
    def convolution2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Implement 2D convolution from scratch.
        
        Args:
            image (np.ndarray): Input image (H, W)
            kernel (np.ndarray): Convolution kernel (Kh, Kw)
        """
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

class SimpleCNN(nn.Module):
    """A basic CNN architecture for demonstration."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and fully connected
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class ResidualBlock(nn.Module):
    """Basic building block for ResNet architecture."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out

class TransferLearningExample:
    """Demonstrates transfer learning with pre-trained models."""
    
    def __init__(self, num_classes: int, model_name: str = 'resnet50'):
        self.model = self.create_transfer_model(num_classes, model_name)
        self.transform = self.get_transform()
    
    @staticmethod
    def create_transfer_model(num_classes: int, model_name: str) -> nn.Module:
        """Create a pre-trained model with modified head."""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            
            # Freeze backbone
            for param in model.parameters():
                param.requires_grad = False
            
            # Modify final layer
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        return model
    
    @staticmethod
    def get_transform() -> transforms.Compose:
        """Get standard transforms for ImageNet models."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def staged_training(self, epochs_per_stage: int):
        """Implement staged training for transfer learning."""
        # Stage 1: Train only new layers
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc.requires_grad = True
        
        # Stage 2: Fine-tune all layers
        for param in self.model.parameters():
            param.requires_grad = True

def main():
    # Example 1: Basic Convolution
    conv_ops = BasicConvolutions()
    edge_kernel = conv_ops.create_edge_detector()
    
    # Create a simple test image
    test_image = np.random.rand(32, 32)
    result = conv_ops.convolution2d(test_image, edge_kernel)
    print("Convolution output shape:", result.shape)
    
    # Example 2: Simple CNN
    model = SimpleCNN()
    sample_input = torch.randn(1, 3, 32, 32)
    output = model(sample_input)
    print("\nCNN output shape:", output.shape)
    
    # Example 3: Transfer Learning
    transfer_model = TransferLearningExample(num_classes=10)
    print("\nTransfer learning model structure:")
    print(transfer_model.model)

if __name__ == "__main__":
    main() 