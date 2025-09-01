import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from datetime import datetime
import shutil

from helpers import get_device, make_writer, run_experiment

# — Hyperparameters —
EPOCHS = 200
LR = 0.01
HIDDEN_DIMS = [32]  # Increased from 16 to 32 for PubMed's larger feature space
DROPOUT = 0.5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 1  # PubMed is a single graph

# Activation function
activation = nn.ReLU; activation_name = "ReLU"

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels, activation_fn, dropout=0.5):
        super().__init__()
        # First layer
        self.conv1 = GCNConv(in_channels, hidden_dims[0])
        self.norm1 = GraphNorm(hidden_dims[0])
        self.activation1 = activation_fn()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second layer
        self.conv2 = GCNConv(hidden_dims[0], out_channels)
        self.norm2 = GraphNorm(out_channels)
        self.activation2 = activation_fn()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        # First layer
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        
        return x

def main():
    # Load Cora dataset
    print("Loading PubMed dataset...")
    dataset = Planetoid(root='data/PubMed', name='PubMed')
    data = dataset[0]
    
    # Create model
    model = GNN(
        in_channels=dataset.num_node_features,
        hidden_dims=HIDDEN_DIMS,
        out_channels=dataset.num_classes,
        activation_fn=activation
    )
    device = get_device(model)
    model = model.to(device)
    data = data.to(device)
    
    # Setup logging
    base_run_dir = 'runs'
    model_run_prefix = f'GNN_PubMed_{activation_name}'
    
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
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
            val_pred = val_out[data.val_mask].argmax(dim=1)
            val_acc = int((val_pred == data.y[data.val_mask]).sum()) / int(data.val_mask.sum())
            
            # Log metrics
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Loss/val', val_loss.item(), epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Test
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_pred = test_out[data.test_mask].argmax(dim=1)
        test_acc = int((test_pred == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())
        print(f'Test Accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main() 