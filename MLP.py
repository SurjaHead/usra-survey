import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datetime import datetime
import os
import shutil
import json
import csv
import pandas as pd
from modal import App, method, Volume, Image

# Define get_device function inline to avoid TensorBoard dependency in helpers.py
def get_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    if not torch.cuda.is_available():
        print("WARNING: No CUDA device found; continuing on CPU.")
    return device

# Note: Data loaders are now created inside the Modal function to ensure they work in the cloud environment

# Define the Modal image with all required dependencies
image = Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scipy"
)

# Create a volume for persistent storage
volume = Volume.from_name("mlp-csv-logs", create_if_missing=True)

app = App("mlp-cuda-timing", image=image)

class SimpleMLP(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, 512)
        self.activation1 = activation_fn()
        self.linear2 = nn.Linear(512, 256)
        self.activation2 = activation_fn()
        self.linear3 = nn.Linear(256, 128)
        self.activation3 = activation_fn()
        self.linear4 = nn.Linear(128, 64)
        self.activation4 = activation_fn()
        self.linear5 = nn.Linear(64, 10)
        
        # Store activation timings for analysis
        self.activation_times = {
            'layer_1': [],
            'layer_2': [],
            'layer_3': [],
            'layer_4': []
        }
        
        # Warmup control
        self.warmup_epochs = 10
        self.current_epoch = 0
        self.timing_enabled = False

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        
        # Time activation 1 (only after warmup)
        if torch.cuda.is_available() and self.timing_enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            x = self.activation1(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            self.activation_times['layer_1'].append(elapsed)
        else:
            x = self.activation1(x)
        
        x = self.linear2(x)
        
        # Time activation 2 (only after warmup)
        if torch.cuda.is_available() and self.timing_enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            x = self.activation2(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            self.activation_times['layer_2'].append(elapsed)
        else:
            x = self.activation2(x)
        
        x = self.linear3(x)
        
        # Time activation 3 (only after warmup)
        if torch.cuda.is_available() and self.timing_enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            x = self.activation3(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            self.activation_times['layer_3'].append(elapsed)
        else:
            x = self.activation3(x)
        
        x = self.linear4(x)
        
        # Time activation 4 (only after warmup)
        if torch.cuda.is_available() and self.timing_enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            x = self.activation4(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            self.activation_times['layer_4'].append(elapsed)
        else:
            x = self.activation4(x)
        
        x = self.linear5(x)
        return x
    
    def get_average_activation_times(self):
        """Return average activation times for each layer"""
        avg_times = {}
        for layer, times in self.activation_times.items():
            if times:
                avg_times[layer] = sum(times) / len(times)
            else:
                avg_times[layer] = 0.0
        return avg_times
    
    def reset_activation_times(self):
        """Reset activation timing storage"""
        for layer in self.activation_times:
            self.activation_times[layer] = []
    
    def set_epoch(self, epoch):
        """Set current epoch and enable timing after warmup"""
        self.current_epoch = epoch
        self.timing_enabled = epoch > self.warmup_epochs

@app.function(gpu="H100", volumes={"/root/csv_logs": volume}, timeout=3600)
def train_mlp_with_timing(activation_fn, activation_name):
    """Train MLP with explicit CUDA timing for each activation layer"""
    # get_device function is now defined globally above
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check GPU configuration.")
    
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 60  # 10 warmup + 50 timing epochs
    LEARNING_RATE = 0.001
    
    # Data Loaders (recreate in the Modal function)
    full_train_dataset = datasets.MNIST('/root/data', train=True, download=True, transform=transforms.ToTensor())
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1000, shuffle=False
    )
    test_loader = DataLoader(
        datasets.MNIST('/root/data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False
    )
    
    print("▶ Script start")
    model = SimpleMLP(activation_fn)
    device = get_device(model)

    print(f"▶ Beginning training for {EPOCHS} epochs…")
    # Create a unique run directory with model and activation name
    model_name = "MLP"
    base_run_dir = '/root/csv_logs'
    model_run_prefix = f'{model_name}_{activation_name}'
    
    # Create base directory if it doesn't exist
    os.makedirs(base_run_dir, exist_ok=True)
    
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
    os.makedirs(run_dir, exist_ok=True)

    # Setup CSV logging
    training_log_path = os.path.join(run_dir, 'training_metrics.csv')
    activation_log_path = os.path.join(run_dir, 'activation_timings.csv')
    
    # Initialize CSV files with headers
    with open(training_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    with open(activation_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'layer_1', 'layer_2', 'layer_3', 'layer_4', 'total_time'])

    # Custom training loop to save activation timing data
    from torch import nn, optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training with explicit activation timing...")
    print(f"🔥 Warmup period: {model.warmup_epochs} epochs (epochs 1-{model.warmup_epochs}, no timing)")
    print(f"📊 Timing period: {EPOCHS - model.warmup_epochs} epochs (epochs {model.warmup_epochs + 1}-{EPOCHS})")
    print(f"📋 CSV timing epochs will be numbered 0-{EPOCHS - model.warmup_epochs - 1}")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.set_epoch(epoch)  # Enable timing after warmup
        
        if model.timing_enabled:
            model.reset_activation_times()  # Only reset timing storage when timing is enabled
        
        total_loss, correct = 0.0, 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        
        # Calculate training metrics
        avg_loss = total_loss / len(train_dataset)
        acc = correct / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss += criterion(logits, y).item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
        
        val_avg_loss = val_loss / len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        # Log training metrics (always log)
        with open(training_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, acc, val_avg_loss, val_acc])
        
        # Get activation timings and log only after warmup
        if model.timing_enabled:
            avg_activation_times = model.get_average_activation_times()
            total_activation_time = sum(avg_activation_times.values())
            
            # Log with timing epoch starting from 0 (actual epoch - warmup epochs)
            timing_epoch = epoch - model.warmup_epochs - 1  # Start from 0
            
            with open(activation_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timing_epoch,
                    avg_activation_times['layer_1'],
                    avg_activation_times['layer_2'],
                    avg_activation_times['layer_3'],
                    avg_activation_times['layer_4'],
                    total_activation_time
                ])
            
            print(f"Epoch {epoch:2d} (Timing {timing_epoch:2d}) | "
                  f"Train {acc*100:5.2f}% ({avg_loss:.4f}) | "
                  f"Val {val_acc*100:5.2f}% ({val_avg_loss:.4f}) | "
                  f"Total Activation Time: {total_activation_time:.3f} ms")
            
            # Print activation timing details
            for layer_name, avg_time in avg_activation_times.items():
                print(f"  {layer_name}: {avg_time:.3f} ms")
        else:
            print(f"Epoch {epoch:2d} | "
                  f"Train {acc*100:5.2f}% ({avg_loss:.4f}) | "
                  f"Val {val_acc*100:5.2f}% ({val_avg_loss:.4f}) | "
                  f"🔥 Warmup ({model.warmup_epochs - epoch + 1} epochs remaining)")
    
    # Final test evaluation
    model.eval()
    test_loss, test_correct = 0.0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            test_loss += criterion(logits, y).item() * X.size(0)
            test_correct += (logits.argmax(1) == y).sum().item()
    
    test_avg_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)
    
    print(f"Final Test | Test {test_acc*100:5.2f}% ({test_avg_loss:.4f})")
    
    # Save final training stats including activation timing
    final_activation_times = model.get_average_activation_times()
    stats = {
        'activation': activation_name,
        'final_train_loss': float(avg_loss),
        'final_train_acc': float(acc),
        'final_val_loss': float(val_avg_loss),
        'final_val_acc': float(val_acc),
        'final_test_loss': float(test_avg_loss),
        'final_test_acc': float(test_acc),
        'activation_times': {k: float(v) for k, v in final_activation_times.items()},
        'total_activation_time': float(sum(final_activation_times.values())),
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'total_epochs': EPOCHS,
            'warmup_epochs': model.warmup_epochs,
            'timing_epochs': EPOCHS - model.warmup_epochs,
            'learning_rate': LEARNING_RATE
        }
    }
    
    with open(os.path.join(run_dir, 'final_results.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print("✔ Training complete")
    return run_dir

@app.local_entrypoint()
def main():
    """Run MLP training with different activation functions on H100"""
    
    # Define activations to test
    activations = [
        (nn.ReLU, "ReLU"),
        (nn.GELU, "GELU"),
        (nn.Sigmoid, "Sigmoid"),
        (nn.Tanh, "Tanh"),
        (nn.ELU, "ELU"),
        (nn.LeakyReLU, "LeakyReLU"),
        (nn.SiLU, "SiLU"),
        (nn.Mish, "Mish"),
    ]
    
    print("Starting MLP training with CUDA activation timing on H100...")
    
    run_dirs = []
    for activation_fn, activation_name in activations:
        print(f"\nTraining MLP with {activation_name} activation...")
        try:
            run_dir = train_mlp_with_timing.remote(activation_fn, activation_name)
            run_dirs.append(run_dir)
            print(f"Completed training MLP with {activation_name}. Logs saved to: {run_dir}")
        except Exception as e:
            print(f"Error training with {activation_name}: {str(e)}")
    
    print(f"\nAll training runs completed! Results saved to {len(run_dirs)} directories.")
    return run_dirs

if __name__ == "__main__":
    # For local testing, just run with one activation
    print("Running locally for testing...")
    model = SimpleMLP(nn.ReLU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Test forward pass with timing enabled (simulate post-warmup)
    model.set_epoch(15)  # Simulate epoch after warmup
    test_input = torch.randn(32, 1, 28, 28).to(device)
    output = model(test_input)
    print(f"Test output shape: {output.shape}")
    print(f"Timing enabled: {model.timing_enabled}")
    print(f"Average activation times: {model.get_average_activation_times()}")
    print("Local test completed successfully!")