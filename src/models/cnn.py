"""Convolutional neural network models."""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network.
    
    Architecture:
        Conv layers -> Pooling -> FC layers -> Output
    """
    
    def __init__(self, num_classes=10, num_conv_layers=2, 
                 num_filters=[32, 64], kernel_size=3, 
                 use_pooling=True, dropout_rate=0.5):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes
            num_conv_layers: Number of convolutional layers
            num_filters: List of number of filters for each conv layer
            kernel_size: Size of convolutional kernel (int or list)
            use_pooling: Whether to use max pooling after conv layers
            dropout_rate: Dropout rate for regularization
        """
        super(SimpleCNN, self).__init__()
        
        self.use_pooling = use_pooling
        
        # Handle single kernel size value
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size] * num_conv_layers
        else:
            kernel_sizes = kernel_size
        
        # Ensure num_filters list matches num_conv_layers
        if len(num_filters) != num_conv_layers:
            raise ValueError("Length of num_filters must match num_conv_layers")
        
        # Build convolutional layers
        conv_layers = []
        in_channels = 3  # RGB input
        
        for i in range(num_conv_layers):
            out_channels = num_filters[i]
            k_size = kernel_sizes[i]
            padding = k_size // 2  # 'same' padding
            
            conv_layers.append(nn.Conv2d(in_channels, out_channels, 
                                        kernel_size=k_size, 
                                        padding=padding))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            
            if use_pooling:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate size after convolutions
        # For CIFAR-10: 32x32 input
        if use_pooling:
            feature_size = 32 // (2 ** num_conv_layers)
        else:
            feature_size = 32
        
        flattened_size = num_filters[-1] * feature_size * feature_size
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_cnn_model(num_conv_layers=2, num_filters=[32, 64], 
                  kernel_size=3, use_pooling=True):
    """
    Get a CNN model instance.
    
    Args:
        num_conv_layers: Number of convolutional layers
        num_filters: List of number of filters for each conv layer
        kernel_size: Size of convolutional kernel
        use_pooling: Whether to use max pooling
        
    Returns:
        SimpleCNN instance
    """
    model = SimpleCNN(
        num_conv_layers=num_conv_layers,
        num_filters=num_filters,
        kernel_size=kernel_size,
        use_pooling=use_pooling
    )
    return model


if __name__ == "__main__":
    # Test the model with different configurations
    
    # Configuration 1: 2 conv layers, 3x3 kernel
    model1 = get_cnn_model(num_conv_layers=2, num_filters=[32, 64], kernel_size=3)
    print("Configuration 1: 2 conv layers, 3x3 kernel")
    print(f"Number of parameters: {model1.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    output = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")
    
    # Configuration 2: 3 conv layers, 5x5 kernel
    model2 = get_cnn_model(num_conv_layers=3, num_filters=[32, 64, 128], kernel_size=5)
    print("Configuration 2: 3 conv layers, 5x5 kernel")
    print(f"Number of parameters: {model2.count_parameters():,}")
