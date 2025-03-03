"""
Week 9: From Supervised to Generative Learning
Code Examples and Implementations

This module contains practical implementations of concepts covered in Week 9:
1. Simple Diffusion Model
2. Masked Prediction Task
3. Contrastive Learning Example
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Part 1: Simple Diffusion Model Implementation
class SimpleDiffusion:
    def __init__(self, num_timesteps=1000):
        """Initialize a basic diffusion model."""
        self.num_timesteps = num_timesteps
        # We'll implement noise scheduling and other components
        pass

    def forward_process(self, x, t):
        """Implement the forward (noise-adding) process."""
        pass

    def reverse_process(self, x, t):
        """Implement the reverse (denoising) process."""
        pass

# Part 2: Masked Prediction Implementation
class MaskedPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize a simple masked prediction model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layer
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, mask):
        """
        Forward pass with masking.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            mask (torch.Tensor): Binary mask of shape [batch_size, input_dim]
                               1 indicates masked (hidden) values
        Returns:
            tuple: (predictions, masked_input)
                  predictions are for masked positions only
        """
        # Create masked input by zeroing out masked positions
        masked_input = x * (1 - mask)
        
        # Encode the masked input
        encoded = self.encoder(masked_input)
        
        # Decode to get predictions
        predictions = self.decoder(encoded)
        
        return predictions, masked_input

def create_random_masks(batch_size, feature_dim, mask_ratio=0.15):
    """
    Create random masks for training.
    
    Args:
        batch_size (int): Number of samples in batch
        feature_dim (int): Dimension of features
        mask_ratio (float): Proportion of values to mask
    """
    mask = torch.zeros(batch_size, feature_dim)
    for i in range(batch_size):
        mask_indices = torch.randperm(feature_dim)[:int(feature_dim * mask_ratio)]
        mask[i, mask_indices] = 1
    return mask

def train_step(model, optimizer, x_batch, mask):
    """
    Single training step.
    
    Args:
        model (MaskedPredictor): The model
        optimizer (torch.optim.Optimizer): The optimizer
        x_batch (torch.Tensor): Batch of input data
        mask (torch.Tensor): Mask indicating which values to predict
    """
    optimizer.zero_grad()
    
    # Forward pass
    predictions, masked_input = model(x_batch, mask)
    
    # Compute loss only for masked values
    loss = F.mse_loss(predictions * mask, x_batch * mask)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example usage
def main():
    # Create synthetic data
    input_dim = 20
    hidden_dim = 64
    batch_size = 32
    
    # Initialize model
    model = MaskedPredictor(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Generate sample data
    x = torch.randn(batch_size, input_dim)
    mask = create_random_masks(batch_size, input_dim)
    
    # Training loop
    for epoch in range(10):
        loss = train_step(model, optimizer, x, mask)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    # Test prediction
    with torch.no_grad():
        predictions, masked_input = model(x, mask)
        # Compare original vs predicted values at masked positions
        masked_original = x[mask.bool()].numpy()
        masked_predicted = predictions[mask.bool()].numpy()
        print("\nSample predictions for masked values:")
        print("Original:", masked_original[:5])
        print("Predicted:", masked_predicted[:5])

if __name__ == "__main__":
    main()

# Part 3: Contrastive Learning Implementation
class ContrastiveLearner(nn.Module):
    def __init__(self, encoder_dim, projection_dim):
        """Initialize a basic contrastive learning setup."""
        super().__init__()
        pass

    def forward(self, x1, x2):
        """Forward pass with positive pairs."""
        pass 