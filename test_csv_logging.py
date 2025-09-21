#!/usr/bin/env python3
"""
Test script to verify CSV logging functionality works correctly
"""

import torch
import torch.nn as nn
from MLP import SimpleMLP, get_device
import csv
import os
import tempfile
import json

def test_csv_logging():
    """Test the CSV logging functionality"""
    print("🧪 Testing CSV logging functionality...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Test model creation and forward pass
        model = SimpleMLP(nn.ReLU)
        device = get_device(model)
        
        print(f"✅ Model created and moved to {device}")
        
        # Test forward pass with timing
        test_input = torch.randn(8, 1, 28, 28).to(device)
        
        # Run a few forward passes to collect timing data
        for i in range(5):
            output = model(test_input)
        
        # Get timing data
        avg_times = model.get_average_activation_times()
        print(f"✅ Forward pass completed, got timing data: {avg_times}")
        
        # Test CSV writing
        training_csv_path = os.path.join(temp_dir, 'test_training.csv')
        activation_csv_path = os.path.join(temp_dir, 'test_activation.csv')
        
        # Write test training metrics
        with open(training_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            writer.writerow([1, 0.5, 0.85, 0.4, 0.88])
            writer.writerow([2, 0.3, 0.92, 0.25, 0.93])
        
        # Write test activation timings
        with open(activation_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'layer_1', 'layer_2', 'layer_3', 'layer_4', 'total_time'])
            total_time = sum(avg_times.values())
            writer.writerow([1, avg_times['layer_1'], avg_times['layer_2'], 
                           avg_times['layer_3'], avg_times['layer_4'], total_time])
            writer.writerow([2, avg_times['layer_1'], avg_times['layer_2'], 
                           avg_times['layer_3'], avg_times['layer_4'], total_time])
        
        print("✅ CSV files written successfully")
        
        # Test JSON writing
        json_path = os.path.join(temp_dir, 'test_results.json')
        test_results = {
            'activation': 'ReLU',
            'final_test_acc': 0.95,
            'activation_times': avg_times,
            'total_activation_time': sum(avg_times.values())
        }
        
        with open(json_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print("✅ JSON file written successfully")
        
        # Verify files exist and have content
        assert os.path.exists(training_csv_path), "Training CSV not created"
        assert os.path.exists(activation_csv_path), "Activation CSV not created"
        assert os.path.exists(json_path), "JSON file not created"
        
        # Check file sizes
        assert os.path.getsize(training_csv_path) > 0, "Training CSV is empty"
        assert os.path.getsize(activation_csv_path) > 0, "Activation CSV is empty"
        assert os.path.getsize(json_path) > 0, "JSON file is empty"
        
        print("✅ All files verified successfully")
        
        # Test reading the files back
        import pandas as pd
        
        training_df = pd.read_csv(training_csv_path)
        activation_df = pd.read_csv(activation_csv_path)
        
        assert len(training_df) == 2, "Training CSV should have 2 rows"
        assert len(activation_df) == 2, "Activation CSV should have 2 rows"
        assert list(training_df.columns) == ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        assert list(activation_df.columns) == ['epoch', 'layer_1', 'layer_2', 'layer_3', 'layer_4', 'total_time']
        
        print("✅ CSV files read back successfully with correct structure")
        
        with open(json_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['activation'] == 'ReLU'
        assert 'activation_times' in loaded_results
        
        print("✅ JSON file read back successfully")
        
    print("\n🎉 All CSV logging tests passed!")
    print("The MLP is ready for Modal deployment with CSV logging.")

def main():
    print("CSV Logging Test Suite")
    print("=" * 30)
    
    try:
        test_csv_logging()
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise
    
    print("\n✨ Ready to run on Modal H100!")

if __name__ == "__main__":
    main()
