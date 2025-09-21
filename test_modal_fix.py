#!/usr/bin/env python3
"""
Quick test to verify the Modal fix works
"""

import torch
import torch.nn as nn

# Test importing the MLP module without helpers dependency
try:
    from MLP import SimpleMLP, get_device
    print("✅ Successfully imported MLP components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

def test_basic_functionality():
    """Test basic MLP functionality without Modal"""
    print("🧪 Testing basic MLP functionality...")
    
    # Create model
    model = SimpleMLP(nn.ReLU)
    device = get_device(model)
    
    print(f"✅ Model created on device: {device}")
    
    # Test forward pass
    test_input = torch.randn(4, 1, 28, 28).to(device)
    output = model(test_input)
    
    print(f"✅ Forward pass successful, output shape: {output.shape}")
    
    # Test timing functionality
    avg_times = model.get_average_activation_times()
    print(f"✅ Activation timing data: {avg_times}")
    
    # Reset timing
    model.reset_activation_times()
    print("✅ Timing reset successful")
    
    return True

def main():
    print("Modal Fix Verification Test")
    print("=" * 30)
    
    try:
        success = test_basic_functionality()
        if success:
            print("\n🎉 All tests passed! The Modal fix should work.")
            print("You can now run: modal run MLP.py")
        else:
            print("\n❌ Tests failed.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
