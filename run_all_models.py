import torch
import torch.nn as nn
from datetime import datetime
import os
import shutil
import time
from MLP import SimpleMLP
from CNN import SimpleCNN
from Transformer import SimpleTransformer
from GNN import GNN
from helpers import get_device, make_writer, run_experiment

# Define all activation functions to test
ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh
}

# Define all models to test
MODELS = {
    "MLP": SimpleMLP,
    "CNN": SimpleCNN,
    "Transformer": SimpleTransformer,
    "GNN": GNN
}

def setup_data_loaders(model_type):
    """Setup appropriate data loaders based on model type"""
    if model_type == "GNN":
        from torch_geometric.datasets import MNISTSuperpixels
        from torch_geometric.loader import DataLoader
        
        # Load MNIST Superpixels dataset
        train_dataset = MNISTSuperpixels(root='data/MNISTSuperpixels', train=True)
        test_dataset = MNISTSuperpixels(root='data/MNISTSuperpixels', train=False)
        
        # Split training data into train and validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        return train_loader, val_loader, test_loader
    else:
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, transforms
        
        # Load MNIST dataset
        full_train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
        test_loader = DataLoader(
            datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
            batch_size=1000, shuffle=False
        )
        
        return train_loader, val_loader, test_loader

def run_experiments(device_type="cuda"):
    """Run all experiments for a specific device type"""
    # Force device type
    if device_type == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    
    print(f"\n{'='*50}")
    print(f"Starting experiments on {device_type.upper()}")
    print(f"{'='*50}\n")
    
    for model_name, model_class in MODELS.items():
        print(f"\nRunning {model_name} experiments...")
        
        # Setup data loaders
        train_loader, val_loader, test_loader = setup_data_loaders(model_name)
        
        for activation_name, activation_fn in ACTIVATIONS.items():
            print(f"\n  Testing {activation_name} activation...")
            
            # Create model
            if model_name == "GNN":
                model = model_class(
                    in_channels=train_loader.dataset[0].num_node_features,
                    hidden_dims=[512, 256, 128, 64],
                    out_channels=10,
                    activation_fn=activation_fn
                )
            else:
                model = model_class(activation_fn)
            
            # Setup device and writer
            device = get_device(model)
            base_run_dir = 'runs'
            model_run_prefix = f'{model_name}_{activation_name}_{device_type.upper()}'
            
            # Remove existing runs
            if os.path.exists(base_run_dir):
                for item in os.listdir(base_run_dir):
                    if item.startswith(model_run_prefix):
                        full_path = os.path.join(base_run_dir, item)
                        print(f"    Removing previous run directory: {full_path}")
                        shutil.rmtree(full_path)
            
            # Create new run directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join(base_run_dir, f"{model_run_prefix}_{timestamp}")
            writer = make_writer(run_dir)
            
            # Run experiment
            try:
                run_experiment(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    device,
                    writer,
                    epochs=50,
                    lr=0.001,
                    is_gnn=(model_name == "GNN")
                )
                print(f"    ✓ Completed {model_name} with {activation_name} on {device_type.upper()}")
            except Exception as e:
                print(f"    ✗ Error in {model_name} with {activation_name}: {str(e)}")
                continue

def main():
    # First run all experiments on GPU
    if torch.cuda.is_available():
        run_experiments("cuda")
    else:
        print("No CUDA device available, skipping GPU experiments")
    
    # Then run all experiments on CPU
    run_experiments("cpu")

if __name__ == "__main__":
    main() 