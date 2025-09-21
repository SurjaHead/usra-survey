import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from datetime import datetime
import shutil
import json
import csv
import modal

from helpers import get_device

# Modal setup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")  # Install torch first
    .pip_install([
        "torch-geometric", 
        "torch-scatter", 
        "torch-sparse", 
        "torch-cluster", 
        "torch-spline-conv",
        "tensorboard"  # Add tensorboard for helpers.py compatibility
    ])  # Then install torch-geometric extensions
    .add_local_python_source("helpers")
)

volume = modal.Volume.from_name("gnn-csv-logs", create_if_missing=True)
app = modal.App("gnn-timing", image=image)

# — Hyperparameters —
WARMUP_EPOCHS = 10
TIMING_EPOCHS = 190
EPOCHS = WARMUP_EPOCHS + TIMING_EPOCHS  # 10 warmup + 190 timing = 200 total
LR = 0.01
HIDDEN_DIMS = [16]  # Changed to match image specification: 1433 -> 16 -> 7
DROPOUT = 0.5
WEIGHT_DECAY = 0.005  # Updated to match image specification
BATCH_SIZE = 1  # Cora only has one graph

# Activation functions for testing
ACTIVATION_FUNCTIONS = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'ELU': nn.ELU,
    'LeakyReLU': nn.LeakyReLU,
    'SiLU': nn.SiLU,
    'Mish': nn.Mish
}

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels, activation_fn, dropout=0.5):
        super().__init__()
        # Architecture based on image specification:
        # GCN Layer 1: GCNConv: 1433 -> 16, GraphNorm, Activation layer 1, Dropout
        # GCN Layer 2: GCNConv: 16 -> 7, GraphNorm, Activation layer 2, Dropout
        
        # First GCN layer
        self.conv1 = GCNConv(in_channels, hidden_dims[0])  # 1433 -> 16
        self.norm1 = GraphNorm(hidden_dims[0])
        self.activation1 = activation_fn()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second GCN layer
        self.conv2 = GCNConv(hidden_dims[0], out_channels)  # 16 -> 7
        self.norm2 = GraphNorm(out_channels)
        self.activation2 = activation_fn()
        self.dropout2 = nn.Dropout(dropout)
        
        # CUDA timing storage
        self.activation_times = {
            'activation1': [],
            'activation2': []
        }
        
        # Warmup and timing control
        self.warmup_epochs = WARMUP_EPOCHS
        self.current_epoch = 0
        self.timing_enabled = False

    def forward(self, x, edge_index, batch=None):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        
        # Time activation layer 1 (only if timing is enabled)
        if self.timing_enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            x = self.activation1(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            self.activation_times['activation1'].append(elapsed_time)
        else:
            x = self.activation1(x)
        
        x = self.dropout1(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        
        # Time activation layer 2 (only if timing is enabled)
        if self.timing_enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            x = self.activation2(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            self.activation_times['activation2'].append(elapsed_time)
        else:
            x = self.activation2(x)
        
        x = self.dropout2(x)
        
        return x
    
    def get_all_activation_times(self):
        """Return all activation times for each layer"""
        return self.activation_times.copy()
    
    def get_average_activation_times(self):
        """Return average activation time for each layer"""
        avg_times = {}
        for layer_name, times in self.activation_times.items():
            if times:
                avg_times[layer_name] = sum(times) / len(times)
            else:
                avg_times[layer_name] = 0.0
        return avg_times
    
    def get_average_total_activation_time_per_forward_pass(self):
        """Return average total activation time per forward pass across the epoch"""
        all_times = self.get_all_activation_times()
        
        # Get the number of forward passes (should be same for all layers)
        num_forward_passes = 0
        for layer_name, times in all_times.items():
            if times:
                num_forward_passes = len(times)
                break
        
        if num_forward_passes == 0:
            return 0.0
        
        # Calculate total activation time for each forward pass
        total_times_per_forward_pass = []
        
        for i in range(num_forward_passes):
            forward_pass_total = 0.0
            for layer_name, times in all_times.items():
                if i < len(times):
                    forward_pass_total += times[i]
            total_times_per_forward_pass.append(forward_pass_total)
        
        # Return average total time per forward pass
        return sum(total_times_per_forward_pass) / len(total_times_per_forward_pass)
    
    def reset_activation_times(self):
        """Reset activation timing storage"""
        for layer_name in self.activation_times:
            self.activation_times[layer_name] = []
    
    def set_epoch(self, epoch):
        """Set current epoch and enable timing after warmup"""
        self.current_epoch = epoch
        self.timing_enabled = epoch >= self.warmup_epochs

@app.function(
    volumes={"/data": volume},
    gpu="H100",
    timeout=7200,  # 2 hours
    scaledown_window=300  # Updated parameter name
)
def train_gnn_with_timing(activation_name: str):
    """Train GNN with CUDA timing and CSV logging"""
    print(f"🚀 Training GNN with {activation_name} activation")
    print(f"📊 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    # Load Cora dataset (changed from PubMed to match batch_size=1 comment)
    print("Loading Cora dataset...")
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    
    print(f"📈 Dataset Info:")
    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.num_edges}")
    print(f"   Features: {dataset.num_node_features}")
    print(f"   Classes: {dataset.num_classes}")
    
    # Get activation function
    activation_fn = ACTIVATION_FUNCTIONS[activation_name]
    
    # Create model
    model = GNN(
        in_channels=dataset.num_node_features,  # 1433 for Cora
        hidden_dims=HIDDEN_DIMS,  # [16]
        out_channels=dataset.num_classes,  # 7 for Cora
        activation_fn=activation_fn,
        dropout=DROPOUT
    )
    
    device = get_device(model)
    model = model.to(device)
    data = data.to(device)
    
    print(f"🔧 Model on device: {device}")
    print(f"🏗️  Architecture: {dataset.num_node_features} → {HIDDEN_DIMS[0]} → {dataset.num_classes}")
    
    # Setup logging directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"GNN_Cora_{activation_name}_{timestamp}"
    
    # Local paths for this run
    local_run_dir = f"/tmp/{run_name}"
    os.makedirs(local_run_dir, exist_ok=True)
    
    # CSV file paths
    training_log_path = os.path.join(local_run_dir, "training_log.csv")
    activation_log_path = os.path.join(local_run_dir, "activation_timings.csv")
    
    # Create CSV headers
    with open(training_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    # Get activation layer names for CSV header
    activation_layer_names = ['activation1', 'activation2']
    
    with open(activation_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch'] + activation_layer_names + ['total_time']
        writer.writerow(header)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🎯 Starting training for {EPOCHS} epochs ({WARMUP_EPOCHS} warmup + {TIMING_EPOCHS} timing)...")
    
    for epoch in range(EPOCHS):
        model.train()
        
        # Set current epoch (enables timing after warmup)
        model.set_epoch(epoch)
        
        # Reset activation timings for this epoch (only matters if timing is enabled)
        if model.timing_enabled:
            model.reset_activation_times()
        
        # Training step
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # This will record activation timings
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        # Calculate training accuracy
        train_pred = out[data.train_mask].argmax(dim=1)
        train_acc = int((train_pred == data.y[data.train_mask]).sum()) / int(data.train_mask.sum())
        
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)  # This will also record activation timings
            val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
            val_pred = val_out[data.val_mask].argmax(dim=1)
            val_acc = int((val_pred == data.y[data.val_mask]).sum()) / int(data.val_mask.sum())
        
        # Log to training CSV (all epochs)
        with open(training_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss.item(), train_acc, val_loss.item(), val_acc])
        
        # Only log activation timings during timing epochs
        if model.timing_enabled:
            # Get activation timings - Average total activation time per forward pass
            total_activation_time = model.get_average_total_activation_time_per_forward_pass()
            avg_activation_times = model.get_average_activation_times()  # Still keep individual averages for detailed logging
            
            # Calculate timing epoch (start from 0 for timing epochs)
            timing_epoch = epoch - model.warmup_epochs
            
            with open(activation_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                timing_row = [timing_epoch]
                for layer_name in activation_layer_names:
                    timing_row.append(avg_activation_times.get(layer_name, 0.0))
                timing_row.append(total_activation_time)
                writer.writerow(timing_row)
        
        # All metrics are already logged to CSV files above
        
        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            if model.timing_enabled:
                timing_epoch = epoch - model.warmup_epochs
                print(f"Epoch {epoch:3d} (Timing {timing_epoch:3d}) | "
                      f"Train {train_acc*100:5.2f}% ({train_loss.item():.4f}) | "
                      f"Val {val_acc*100:5.2f}% ({val_loss.item():.4f}) | "
                      f"Avg Total Activation Time/Forward Pass: {total_activation_time:.3f} ms")
            else:
                print(f"Epoch {epoch:3d} (Warmup) | "
                      f"Train {train_acc*100:5.2f}% ({train_loss.item():.4f}) | "
                      f"Val {val_acc*100:5.2f}% ({val_loss.item():.4f})")
    
    # Final test
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_pred = test_out[data.test_mask].argmax(dim=1)
        test_acc = int((test_pred == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())
    
    print(f"\n🎯 Final Test Accuracy: {test_acc:.4f}")
    
    # Get final timing data if available
    final_timing_data = {}
    if model.timing_enabled:
        final_total_activation_time = model.get_average_total_activation_time_per_forward_pass()
        final_activation_times = model.get_average_activation_times()
        final_timing_data = {
            'final_total_activation_time': float(final_total_activation_time),
            'final_activation_times': {k: float(v) for k, v in final_activation_times.items()}
        }
    
    # Save final results
    final_results = {
        'model': 'GNN',
        'dataset': 'Cora',
        'activation': activation_name,
        'test_accuracy': float(test_acc),
        'final_train_accuracy': float(train_acc),
        'final_val_accuracy': float(val_acc),
        'final_train_loss': float(train_loss.item()),
        'final_val_loss': float(val_loss.item()),
        'total_epochs': EPOCHS,
        'warmup_epochs': WARMUP_EPOCHS,
        'timing_epochs': TIMING_EPOCHS,
        'learning_rate': LR,
        'weight_decay': WEIGHT_DECAY,
        'dropout': DROPOUT,
        'hidden_dims': HIDDEN_DIMS,
        'architecture': f"{dataset.num_node_features} → {HIDDEN_DIMS[0]} → {dataset.num_classes}",
        'timestamp': timestamp,
        **final_timing_data
    }
    
    with open(os.path.join(local_run_dir, "final_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Copy results to Modal volume
    volume_path = f"/data/{run_name}"
    print(f"📁 Copying results to Modal volume: {volume_path}")
    
    import shutil
    shutil.copytree(local_run_dir, volume_path)
    
    print(f"✅ Training complete! Results saved to {volume_path}")
    
    return {
        'run_name': run_name,
        'test_accuracy': test_acc,
        'final_activation_time': total_activation_time
    }

@app.local_entrypoint()
def main():
    """Run GNN training for all activation functions"""
    print("🚀 Starting GNN activation function comparison on Modal")
    
    results = []
    for activation_name in ACTIVATION_FUNCTIONS.keys():
        print(f"\n{'='*60}")
        print(f"🧠 Training with {activation_name} activation")
        print('='*60)
        
        result = train_gnn_with_timing.remote(activation_name)
        results.append(result)
        
        print(f"✅ Completed {activation_name}: Test Acc = {result['test_accuracy']:.4f}, "
              f"Activation Time = {result['final_activation_time']:.3f}ms")
    
    print(f"\n🎉 All training completed!")
    print("\n📊 Summary:")
    for result in results:
        print(f"  {result['run_name']}: {result['test_accuracy']:.4f} acc, {result['final_activation_time']:.3f}ms")

# For local testing
if __name__ == "__main__":
    # Test locally with ReLU
    print("🧪 Local testing with ReLU activation")
    activation_name = "ReLU"
    activation_fn = ACTIVATION_FUNCTIONS[activation_name]
    
    # Load Cora dataset
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    
    # Create model
    model = GNN(
        in_channels=dataset.num_node_features,
        hidden_dims=HIDDEN_DIMS,
        out_channels=dataset.num_classes,
        activation_fn=activation_fn,
        dropout=DROPOUT
    )
    
    device = get_device(model)
    model = model.to(device)
    data = data.to(device)
    
    print(f"✅ Model created successfully!")
    print(f"📊 Architecture: {dataset.num_node_features} → {HIDDEN_DIMS[0]} → {dataset.num_classes}")
    print(f"🎮 Device: {device}")
    
    # Test forward pass with timing enabled (simulate post-warmup)
    model.set_epoch(WARMUP_EPOCHS)  # Enable timing
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        print(f"🔍 Output shape: {out.shape}")
        
        # Check activation timing
        total_time = model.get_average_total_activation_time_per_forward_pass()
        avg_times = model.get_average_activation_times()
        print(f"⏱️  Total activation time: {total_time:.3f}ms")
        for layer, time in avg_times.items():
            print(f"   {layer}: {time:.3f}ms") 