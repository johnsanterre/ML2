"""
Week 9: From Supervised to Generative Learning
Code Examples and Implementations - TensorFlow Version

This example focuses on masked prediction, a key self-supervised learning technique.
We'll implement a simple version that can work with numerical feature vectors.
"""

import tensorflow as tf
import numpy as np

class MaskedPredictor(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize a simple masked prediction model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layer
        """
        super().__init__()
        
        # Encoder layers
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu')
        ])
        
        # Decoder layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(input_dim)
        ])

    def call(self, inputs, mask, training=False):
        """
        Forward pass with masking.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape [batch_size, input_dim]
            mask (tf.Tensor): Binary mask of shape [batch_size, input_dim]
                            1 indicates masked (hidden) values
            training (bool): Whether in training mode
        Returns:
            tuple: (predictions, masked_input)
        """
        # Create masked input by zeroing out masked positions
        masked_input = inputs * (1.0 - mask)
        
        # Encode the masked input
        encoded = self.encoder(masked_input, training=training)
        
        # Decode to get predictions
        predictions = self.decoder(encoded, training=training)
        
        return predictions, masked_input

def create_random_masks(batch_size, feature_dim, mask_ratio=0.15):
    """
    Create random masks for training.
    
    Args:
        batch_size (int): Number of samples in batch
        feature_dim (int): Dimension of features
        mask_ratio (float): Proportion of values to mask
    """
    masks = []
    num_mask = int(feature_dim * mask_ratio)
    
    for _ in range(batch_size):
        mask = np.zeros(feature_dim)
        mask_indices = np.random.choice(feature_dim, num_mask, replace=False)
        mask[mask_indices] = 1
        masks.append(mask)
    
    return tf.convert_to_tensor(masks, dtype=tf.float32)

@tf.function
def train_step(model, optimizer, x_batch, mask):
    """
    Single training step.
    
    Args:
        model (MaskedPredictor): The model
        optimizer (tf.keras.optimizers.Optimizer): The optimizer
        x_batch (tf.Tensor): Batch of input data
        mask (tf.Tensor): Mask indicating which values to predict
    """
    with tf.GradientTape() as tape:
        # Forward pass
        predictions, masked_input = model(x_batch, mask, training=True)
        
        # Compute loss only for masked values
        loss = tf.reduce_mean(
            tf.square((predictions - x_batch) * mask)
        )
    
    # Compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create synthetic data
    input_dim = 20
    hidden_dim = 64
    batch_size = 32
    
    # Initialize model
    model = MaskedPredictor(input_dim, hidden_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Generate sample data
    x = tf.random.normal([batch_size, input_dim])
    mask = create_random_masks(batch_size, input_dim)
    
    # Training loop
    for epoch in range(10):
        loss = train_step(model, optimizer, x, mask)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
    
    # Test prediction
    predictions, masked_input = model(x, mask, training=False)
    
    # Compare original vs predicted values at masked positions
    mask_bool = tf.cast(mask, tf.bool)
    masked_original = tf.boolean_mask(x, mask_bool)
    masked_predicted = tf.boolean_mask(predictions, mask_bool)
    
    print("\nSample predictions for masked values:")
    print("Original:", masked_original[:5].numpy())
    print("Predicted:", masked_predicted[:5].numpy())

if __name__ == "__main__":
    main() 