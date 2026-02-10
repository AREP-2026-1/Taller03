"""Baseline fully connected neural network model."""

import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """
    Baseline fully connected neural network without convolutional layers.
    
    Architecture:
        Input (3x32x32) -> Flatten (3072) -> FC layers -> Output (10)
    """
    
    def __init__(self, input_size=3*32*32, hidden_sizes=[512, 256], num_classes=10):
        """
        Initialize the baseline model.
        
        Args:
            input_size: Size of flattened input (channels * height * width)
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
        """
        super(BaselineModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten the input
        x = x.view(x.size(0), -1)
        return self.network(x)
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_baseline_model(hidden_sizes=[512, 256]):
    """
    Get a baseline model instance.
    
    Args:
        hidden_sizes: List of hidden layer sizes
        
    Returns:
        BaselineModel instance
    """
    model = BaselineModel(hidden_sizes=hidden_sizes)
    return model


if __name__ == "__main__":
    # Test the model
    model = get_baseline_model()
    print(f"Model: {model}")
    print(f"Number of parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
