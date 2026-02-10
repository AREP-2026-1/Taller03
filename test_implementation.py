"""Test script to verify the implementation structure."""

import sys
sys.path.insert(0, 'src')

def test_models():
    """Test that models can be imported and instantiated."""
    from models.baseline import get_baseline_model
    from models.cnn import get_cnn_model
    
    # Test baseline model
    baseline = get_baseline_model(hidden_sizes=[512, 256])
    assert baseline.count_parameters() > 0, "Baseline model has no parameters"
    print(f"✓ Baseline model: {baseline.count_parameters():,} parameters")
    
    # Test CNN model with different configurations
    cnn1 = get_cnn_model(num_conv_layers=2, num_filters=[32, 64], kernel_size=3)
    assert cnn1.count_parameters() > 0, "CNN model has no parameters"
    print(f"✓ CNN (2 layers, 3x3 kernel): {cnn1.count_parameters():,} parameters")
    
    cnn2 = get_cnn_model(num_conv_layers=3, num_filters=[32, 64, 128], kernel_size=5)
    print(f"✓ CNN (3 layers, 5x5 kernel): {cnn2.count_parameters():,} parameters")
    
    print("\n✓ All models working correctly")

def test_utils():
    """Test that utility functions can be imported."""
    from utils.data_loader import get_class_names
    from utils.visualization import plot_training_history
    
    class_names = get_class_names()
    assert len(class_names) == 10, "CIFAR-10 should have 10 classes"
    print(f"✓ Class names: {class_names}")
    
    print("✓ All utilities working correctly")

def test_trainer():
    """Test that training utilities can be imported."""
    from training.trainer import train_epoch, validate, train_model, evaluate_model
    
    print("✓ Training utilities imported successfully")

def main():
    print("="*60)
    print("Testing CNN Homework Implementation")
    print("="*60)
    print()
    
    print("1. Testing Models...")
    test_models()
    print()
    
    print("2. Testing Utilities...")
    test_utils()
    print()
    
    print("3. Testing Training Utilities...")
    test_trainer()
    print()
    
    print("="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print()
    print("Implementation is ready!")
    print("To use:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run notebooks in order:")
    print("     - notebooks/01_eda.ipynb")
    print("     - notebooks/02_baseline.ipynb")
    print("     - notebooks/03_cnn.ipynb")

if __name__ == "__main__":
    main()
