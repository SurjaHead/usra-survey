import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
from datetime import datetime
import shutil

from helpers import get_device, make_writer, run_experiment

# — Hyperparameters —
EPOCHS = 50
LR = 0.001
HIDDEN_DIMS = [512, 256, 128, 64]  # Four hidden layers to match MLP's 5 total layers
BATCH_SIZE = 64

# Activation function
# Uncomment ONE of the following lines to choose the activation function:
# activation = nn.ReLU; activation_name = "ReLU"
# activation = nn.GELU; activation_name = "GELU"
# activation = nn.Sigmoid; activation_name = "Sigmoid"
activation = nn.Tanh; activation_name = "Tanh"

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels, activation_fn):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dims[0])
        self.norm1 = GraphNorm(hidden_dims[0])
        self.activation1 = activation_fn()
        
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.norm2 = GraphNorm(hidden_dims[1])
        self.activation2 = activation_fn()
        
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.norm3 = GraphNorm(hidden_dims[2])
        self.activation3 = activation_fn()
        
        self.conv4 = GCNConv(hidden_dims[2], hidden_dims[3])
        self.norm4 = GraphNorm(hidden_dims[3])
        self.activation4 = activation_fn()
        
        # Final layer for graph-level classification
        self.linear = nn.Linear(hidden_dims[3], out_channels)

    def forward(self, x, edge_index, batch=None):
        # First layer
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.activation1(x)
        
        # Second layer
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.activation2(x)
        
        # Third layer
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = self.activation3(x)
        
        # Fourth layer
        x = self.conv4(x, edge_index)
        x = self.norm4(x)
        x = self.activation4(x)
        
        # Global pooling to get graph-level features
        x = global_mean_pool(x, batch)
        
        # Output layer
        x = self.linear(x)
        return x

def main():
    # Load MNIST Superpixels dataset
    print("Loading MNIST Superpixels dataset...")
    train_dataset = MNISTSuperpixels(root='data/MNISTSuperpixels', train=True)
    test_dataset = MNISTSuperpixels(root='data/MNISTSuperpixels', train=False)
    
    # Split training data into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    model = GNN(
        in_channels=train_dataset[0].num_node_features,  # Features per superpixel
        hidden_dims=HIDDEN_DIMS,
        out_channels=10,  # 10 classes for MNIST
        activation_fn=activation
    )
    device = get_device(model)
    
    # Setup logging
    base_run_dir = 'runs'
    model_run_prefix = f'GNN_{activation_name}'
    
    # Remove existing runs
    if os.path.exists(base_run_dir):
        for item in os.listdir(base_run_dir):
            if item.startswith(model_run_prefix):
                full_path = os.path.join(base_run_dir, item)
                print(f"Removing previous run directory: {full_path}")
                shutil.rmtree(full_path)
    
    # Create new run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_run_dir, f"{model_run_prefix}_{timestamp}")
    writer = make_writer(run_dir)
    
    # Run experiment
    run_experiment(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        writer,
        epochs=EPOCHS,
        lr=LR,
        is_gnn=True
    )

if __name__ == "__main__":
    main() 