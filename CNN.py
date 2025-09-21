import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from datetime import datetime
import os
import shutil
import json
import csv
from modal import App, method, Volume, Image

from helpers import get_device, make_writer, train_one_epoch, eval_on, log_metrics

# Define the Modal image with all required dependencies
image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision", 
        "numpy",
        "pandas",
        "scipy",
        "tensorboard"
    )
    .add_local_python_source("helpers")
)

# Create a volume for persistent storage
volume = Volume.from_name("cnn-csv-logs", create_if_missing=True)

app = App("cnn-cuda-timing", image=image)

# — Hyperparameters —
BATCH_SIZE = 128
EPOCHS = 90
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# — Data Transforms —
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010]
)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# — Data Loaders —
train_dataset = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=train_transform
)

val_dataset = datasets.CIFAR10(
    root='data',
    train=False,
    download=True,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, activation_fn=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = activation_fn()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = activation_fn()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Store activation timings for analysis
        self.activation_times = {
            'activation1': [],
            'activation2': []
        }

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        
        # Time first activation
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            out = self.activation1(out)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            self.activation_times['activation1'].append(elapsed)
        else:
            out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        
        # Time second activation
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            out = self.activation2(out)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            self.activation_times['activation2'].append(elapsed)
        else:
            out = self.activation2(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, activation_fn=nn.ReLU):
        super().__init__()
        self.in_channels = 64
        self.activation_fn = activation_fn

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Changed for CIFAR-10
        self.bn1 = nn.BatchNorm2d(64)
        self.main_activation = activation_fn()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Store activation timings for analysis
        self.activation_times = {
            'main_activation': []
        }

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.activation_fn))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, activation_fn=self.activation_fn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Time main activation
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            x = self.main_activation(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            self.activation_times['main_activation'].append(elapsed)
        else:
            x = self.main_activation(x)
            
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def get_all_activation_times(self):
        """Collect activation times from all BasicBlocks and main ResNet"""
        all_times = {}
        
        # Get main activation times
        all_times.update(self.activation_times)
        
        # Get times from all BasicBlocks in all layers
        block_count = 0
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, layer_name)
            for i, block in enumerate(layer):
                if hasattr(block, 'activation_times'):
                    for activation_name, times in block.activation_times.items():
                        key = f'{layer_name}_block{i}_{activation_name}'
                        all_times[key] = times
                        block_count += 1
        
        return all_times
    
    def get_average_activation_times(self):
        """Return average activation times for all layers"""
        all_times = self.get_all_activation_times()
        avg_times = {}
        
        for layer_name, times in all_times.items():
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
        """Reset activation timing storage for all layers"""
        # Reset main activation times
        for layer in self.activation_times:
            self.activation_times[layer] = []
        
        # Reset BasicBlock activation times
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, layer_name)
            for block in layer:
                if hasattr(block, 'activation_times'):
                    for activation_layer in block.activation_times:
                        block.activation_times[activation_layer] = []

# Define get_device function inline for Modal compatibility
def get_device_inline(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    if not torch.cuda.is_available():
        print("WARNING: No CUDA device found; continuing on CPU.")
    return device

@app.function(gpu="H100", volumes={"/root/csv_logs": volume}, timeout=7200)
def train_cnn_with_timing(activation_fn, activation_name):
    """Train CNN with explicit CUDA timing for each activation layer"""
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check GPU configuration.")
    
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Recreate data loaders in Modal function
    train_dataset = datasets.CIFAR10(
        root='/root/data',
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.CIFAR10(
        root='/root/data',
        train=False,
        download=True,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print("▶ Script start")
    
    # Create ResNet model with specified activation
    model = ResNet(BasicBlock, [2, 2, 2, 2], activation_fn=activation_fn)  # ResNet-18
    device = get_device_inline(model)
    model = model.to(device)

    print(f"▶ Beginning training for {EPOCHS} epochs…")
    # Create a unique run directory with model and activation name
    model_name = "ResNet18"
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
    
    # Initialize activation timing CSV with dynamic headers
    model.reset_activation_times()
    # Do a dummy forward pass to get layer names
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    _ = model(dummy_input)
    activation_layer_names = list(model.get_average_activation_times().keys())
    model.reset_activation_times()
    
    with open(activation_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['epoch'] + activation_layer_names + ['total_time']
        writer.writerow(headers)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("Starting training with explicit activation timing...")
    
    # Training loop with activation timing
    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.reset_activation_times()  # Reset timing storage each epoch
        
        total_loss, correct, total_samples = 0.0, 0, 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total_samples += X.size(0)
        
        # Calculate training metrics
        avg_loss = total_loss / total_samples
        acc = correct / total_samples
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss += criterion(logits, y).item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += X.size(0)
        
        val_avg_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Get activation timings - Average total activation time per forward pass
        total_activation_time = model.get_average_total_activation_time_per_forward_pass()
        avg_activation_times = model.get_average_activation_times()  # Still keep individual averages for detailed logging
        
        # Log to CSV files
        with open(training_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, acc, val_avg_loss, val_acc])
        
        with open(activation_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            timing_row = [epoch]
            for layer_name in activation_layer_names:
                timing_row.append(avg_activation_times.get(layer_name, 0.0))
            timing_row.append(total_activation_time)
            writer.writerow(timing_row)
        
        print(f"Epoch {epoch:2d} | "
              f"Train {acc*100:5.2f}% ({avg_loss:.4f}) | "
              f"Val {val_acc*100:5.2f}% ({val_avg_loss:.4f}) | "
              f"Avg Total Activation Time/Forward Pass: {total_activation_time:.3f} ms")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
    
    # Final validation evaluation
    model.eval()
    final_val_loss, final_val_correct, final_val_total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            final_val_loss += criterion(logits, y).item() * X.size(0)
            final_val_correct += (logits.argmax(1) == y).sum().item()
            final_val_total += X.size(0)
    
    final_val_avg_loss = final_val_loss / final_val_total
    final_val_acc = final_val_correct / final_val_total
    
    print(f"Final Validation | Val {final_val_acc*100:5.2f}% ({final_val_avg_loss:.4f})")
    
    # Save final training stats including activation timing
    final_activation_times = model.get_average_activation_times()
    final_avg_total_time = model.get_average_total_activation_time_per_forward_pass()
    stats = {
        'activation': activation_name,
        'final_train_loss': float(avg_loss),
        'final_train_acc': float(acc),
        'final_val_loss': float(final_val_avg_loss),
        'final_val_acc': float(final_val_acc),
        'activation_times': {k: float(v) for k, v in final_activation_times.items()},
        'avg_total_activation_time_per_forward_pass': float(final_avg_total_time),
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'momentum': MOMENTUM,
            'weight_decay': WEIGHT_DECAY
        }
    }
    
    with open(os.path.join(run_dir, 'final_results.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print("✔ Training complete")
    return run_dir

@app.local_entrypoint()
def main():
    """Run CNN training with different activation functions on H100"""
    
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
    
    print("Starting CNN training with CUDA activation timing on H100...")
    
    run_dirs = []
    for activation_fn, activation_name in activations:
        print(f"\nTraining CNN with {activation_name} activation...")
        try:
            run_dir = train_cnn_with_timing.remote(activation_fn, activation_name)
            run_dirs.append(run_dir)
            print(f"Completed training CNN with {activation_name}. Logs saved to: {run_dir}")
        except Exception as e:
            print(f"Error training with {activation_name}: {str(e)}")
    
    print(f"\nAll training runs completed! Results saved to {len(run_dirs)} directories.")
    return run_dirs

if __name__ == "__main__":
    # For local testing, just run with one activation
    print("Running locally for testing...")
    model = ResNet(BasicBlock, [2, 2, 2, 2], activation_fn=nn.ReLU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Test forward pass with timing
    test_input = torch.randn(1, 3, 32, 32).to(device)
    output = model(test_input)
    print(f"Test output shape: {output.shape}")
    print(f"Average activation times: {model.get_average_activation_times()}")
    print("Local test completed successfully!")
