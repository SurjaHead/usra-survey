import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool
from torch_geometric.utils import dropout_edge
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import shutil
from torch_geometric.loader import DataLoader

from helpers import (
    get_device,
    make_writer,
    log_metrics,
    run_experiment
)

# — Hyperparameters —
EPOCHS = 10
LR = 0.001
WEIGHT_DECAY = 5e-4
HIDDEN_DIMS = [32, 16, 8]  # Three hidden layers
DROPOUT_RATE = 0.4
BATCH_SIZE = 64

# Activation function (uncomment one)
# activation = nn.ReLU; activation_name = "ReLU"
# activation = nn.GELU; activation_name = "GELU"
# activation = nn.Sigmoid; activation_name = "Sigmoid"
# activation = nn.ELU; activation_name = "ELU"
# activation = nn.LeakyReLU; activation_name = "LeakyReLU"
activation = nn.Tanh; activation_name = "Tanh"

# — Data Loading —
dataset = MNISTSuperpixels(root='data/MNISTSuperpixels')
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
test_dataset = MNISTSuperpixels(root='data/MNISTSuperpixels', train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# — Model Definition —
class SimpleGNN(nn.Module):
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
        
        self.linear = nn.Linear(hidden_dims[2], out_channels)

    def forward(self, x, edge_index, batch):
        # First layer
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.activation1(x)
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.activation2(x)
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        
        # Third layer
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = self.activation3(x)
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        # Output layer
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

# — Main Experiment Loop —
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGNN(
        in_channels=dataset.num_node_features,
        hidden_dims=HIDDEN_DIMS,
        out_channels=10,  # 10 digits
        activation_fn=activation
    ).to(device)

    # Setup logging with activation name in the run directory
    base_run_dir = 'runs'
    model_run_prefix = f'GNN_{activation_name}'
    
    # Remove all existing runs with the same model and activation
    if os.path.exists(base_run_dir):
        for item in os.listdir(base_run_dir):
            if item.startswith(model_run_prefix):
                full_path = os.path.join(base_run_dir, item)
                print(f"Removing previous run directory: {full_path}")
                shutil.rmtree(full_path)
    
    # Create new run directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_run_dir, f"{model_run_prefix}_{timestamp}")
    writer = make_writer(run_dir)

    # Run experiment with GNN flag
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
