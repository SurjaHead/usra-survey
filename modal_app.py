from modal import Image, App, method, Volume
import torch
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
import torch.nn as nn

# Define the Modal image with all required dependencies
image = Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "tensorboard",
    "torch-geometric",
    "numpy",
    "pandas",
    "scipy"
).add_local_dir(".", remote_path="/root")

# Create a volume for persistent storage
volume = Volume.from_name("tensorboard-logs", create_if_missing=True)

app = App("usra-survey", image=image)

@app.function(gpu="H100", volumes={"/root/tensorboard_logs": volume}, timeout=3600)
def train_cnn(activation_fn, activation_name):
    # Import your training code
    from CNN import ResNet, BasicBlock
    from helpers import get_device, make_writer, train_one_epoch, eval_on, log_metrics
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from datetime import datetime
    import os
    import shutil
    import json

    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check GPU configuration.")
    
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 90
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    # Data Transforms
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

    # Data Loaders
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

    # Create ResNet model
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)  # ResNet-18 for CIFAR-10
    device = get_device(model)
    model = model.to(device)

    # Replace all activation functions with the specified activation
    def replace_activations(module):
        for name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.LeakyReLU)):
                # Create a new instance of the activation function
                if activation_fn in (nn.ReLU, nn.LeakyReLU, nn.ELU):
                    # These activations support inplace
                    new_activation = activation_fn(inplace=getattr(child, 'inplace', False))
                else:
                    # These activations don't support inplace
                    new_activation = activation_fn()
                setattr(module, name, new_activation)
            else:
                replace_activations(child)
    
    replace_activations(model)

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

    # Setup logging
    base_run_dir = '/root/tensorboard_logs'
    model_run_prefix = f'ResNet18_{activation_name}'
    
    # Create base directory if it doesn't exist
    os.makedirs(base_run_dir, exist_ok=True)
    
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
    os.makedirs(run_dir, exist_ok=True)
    writer = make_writer(run_dir)

    # Training loop
    best_acc = 0
    start_epoch = 0
    
    # Check for checkpoint
    checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        train_loss, train_acc, activation_timings = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, epoch)
        val_loss, val_acc = eval_on(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        log_metrics(writer, epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Sum up all activation times for this epoch
        total_activation_time = 0
        for name, times in activation_timings.items():
            if times:  # Only sum if we have measurements
                avg_time = sum(times) / len(times)
                total_activation_time += avg_time
        
        # Log total activation time
        writer.add_scalar('Activation_Time/Total', total_activation_time, epoch)
        print(f'Total Activation Time: {total_activation_time:.3f} ms')
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
        
        # Ensure logs are written
        writer.flush()

    # Save final training stats
    stats = {
        'final_train_loss': float(train_loss),
        'final_train_acc': float(train_acc),
        'final_val_loss': float(val_loss),
        'final_val_acc': float(val_acc),
        'best_val_acc': float(best_acc),
        'activation': activation_name
    }
    with open(os.path.join(run_dir, 'training_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    # Close the writer
    writer.close()
    print(f"Training complete for {activation_name}")
    return run_dir

@app.function(gpu="H100", volumes={"/root/tensorboard_logs": volume}, timeout=3600)
def train_gnn(activation_fn, activation_name):
    # Import your training code
    from GNN import GNN
    from helpers import get_device, make_writer, train_one_epoch, eval_on, log_metrics
    from torch_geometric.datasets import Planetoid
    from torch_geometric.loader import DataLoader
    import torch.nn as nn
    import os
    import shutil
    import json
    import numpy as np

    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check GPU configuration.")
    
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    # Hyperparameters from original GCN paper
    EPOCHS = 200
    LR = 0.01
    HIDDEN_DIMS = [16]  # 16 hidden units as per paper
    DROPOUT = 0.5  # 0.5 dropout rate as per paper
    WEIGHT_DECAY = 5e-4  # L2 regularization as per paper
    BATCH_SIZE = 1  # Cora is a single graph

    # Load Cora dataset
    print("Loading PubMed dataset...")
    dataset = Planetoid(root='/root/data/PubMed', name='PubMed')
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

    # Loss and optimizer with L2 regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Setup logging
    base_run_dir = '/root/tensorboard_logs'
    model_run_prefix = f'GNN_PubMed_{activation_name}'
    
    # Create base directory if it doesn't exist
    os.makedirs(base_run_dir, exist_ok=True)
    
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
    os.makedirs(run_dir, exist_ok=True)
    writer = make_writer(run_dir)

    # Training loop
    best_acc = 0
    start_epoch = 0
    
    # Check for checkpoint
    checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        # Use train_one_epoch for proper activation timing
        train_loss, train_acc, activation_timings = train_one_epoch(
            model, data, criterion, optimizer, device, writer, epoch, is_gnn=True
        )
        
        # Validation
        val_loss, val_acc = eval_on(model, data, criterion, device, is_gnn=True)
        
        # Log metrics
        log_metrics(writer, epoch, train_loss, train_acc, val_loss, val_acc)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
        
        # Ensure logs are written
        writer.flush()

    # Test
    test_loss, test_acc = eval_on(model, data, criterion, device, is_gnn=True)
    print(f'Test Accuracy: {test_acc:.4f}')

    # Save final training stats
    stats = {
        'final_train_loss': float(train_loss),
        'final_val_loss': float(val_loss),
        'final_val_acc': float(val_acc),
        'final_test_acc': float(test_acc),
        'best_val_acc': float(best_acc),
        'epochs_trained': int(epoch + 1),
        'activation': activation_name,
        'hyperparameters': {
            'hidden_units': int(HIDDEN_DIMS[0]),
            'dropout': float(DROPOUT),
            'weight_decay': float(WEIGHT_DECAY),
            'learning_rate': float(LR)
        }
    }
    with open(os.path.join(run_dir, 'training_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    # Close the writer
    writer.close()
    print(f"Training complete for {activation_name}")
    return run_dir

@app.function(gpu="H100", volumes={"/root/tensorboard_logs": volume}, timeout=3600)
def train_mlp(activation_fn, activation_name):
    # Import your training code
    from helpers import get_device, make_writer, train_one_epoch, eval_on, log_metrics
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import os
    import shutil
    import json

    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check GPU configuration.")
    
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.001
    HIDDEN_DIMS = [2084, 2084, 512]  # Three hidden layers

    # Data Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Data Loaders
    train_dataset = datasets.MNIST(
        root='/root/data',
        train=True,
        download=True,
        transform=transform
    )

    val_dataset = datasets.MNIST(
        root='/root/data',
        train=False,
        download=True,
        transform=transform
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

    # Create MLP model
    class MLP(nn.Module):
        def __init__(self, activation_fn):
            super().__init__()
            self.flatten = nn.Flatten()
            self.layers = nn.Sequential(
                nn.Linear(28 * 28, HIDDEN_DIMS[0]),  # 784 -> 2084
                activation_fn(),
                nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),  # 2084 -> 2084
                activation_fn(),
                nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2]),  # 2084 -> 512
                activation_fn(),
                nn.Linear(HIDDEN_DIMS[2], 10)  # 512 -> 10
            )

        def forward(self, x):
            x = self.flatten(x)
            return self.layers(x)

    model = MLP(activation_fn)
    device = get_device(model)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Setup logging
    base_run_dir = '/root/tensorboard_logs'
    model_run_prefix = f'MLP_MNIST_{activation_name}'
    
    # Create base directory if it doesn't exist
    os.makedirs(base_run_dir, exist_ok=True)
    
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
    os.makedirs(run_dir, exist_ok=True)
    writer = make_writer(run_dir)

    # Training loop
    best_acc = 0
    start_epoch = 0
    
    # Check for checkpoint
    checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        train_loss, train_acc, activation_timings = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, epoch)
        val_loss, val_acc = eval_on(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        log_metrics(writer, epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Sum up all activation times for this epoch
        total_activation_time = 0
        for name, times in activation_timings.items():
            if times:  # Only sum if we have measurements
                avg_time = sum(times) / len(times)
                total_activation_time += avg_time
        
        # Log total activation time
        writer.add_scalar('Activation_Time/Total', total_activation_time, epoch)
        print(f'Total Activation Time: {total_activation_time:.3f} ms')
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
        
        # Ensure logs are written
        writer.flush()

    # Save final training stats
    stats = {
        'final_train_loss': float(train_loss),
        'final_train_acc': float(train_acc),
        'final_val_loss': float(val_loss),
        'final_val_acc': float(val_acc),
        'best_val_acc': float(best_acc),
        'activation': activation_name,
        'hyperparameters': {
            'hidden_dims': HIDDEN_DIMS,
            'learning_rate': float(LEARNING_RATE),
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS
        }
    }
    with open(os.path.join(run_dir, 'training_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    # Close the writer
    writer.close()
    print(f"Training complete for {activation_name}")
    return run_dir

def start_tensorboard(log_dir):
    """Start TensorBoard in the background"""
    try:
        # Create local log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Start TensorBoard process
        process = subprocess.Popen(
            ["tensorboard", "--logdir", log_dir, "--port", "6006"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"\nTensorBoard started at http://localhost:6006")
        print(f"Logs will be saved to: {log_dir}")
        return process
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")
        return None

@app.local_entrypoint()
def main():
    # Create local log directory
    local_log_dir = Path("tensorboard_logs")
    local_log_dir.mkdir(exist_ok=True)
    
    # Start TensorBoard
    tb_process = start_tensorboard(str(local_log_dir))
    
    try:
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
        
        # Define models to train
        models = {
            # 'cnn': train_cnn,
            # 'gnn': train_gnn,
            'mlp': train_mlp
        }
        
        # Train with each activation for each model
        run_dirs = []
        for model_name, train_fn in models.items():
            print(f"\nTraining {model_name.upper()} models...")
            for activation_fn, activation_name in activations:
                print(f"\nTraining {model_name.upper()} with {activation_name} activation...")
                run_dir = train_fn.remote(activation_fn, activation_name)
                run_dirs.append(run_dir)
                print(f"Completed training {model_name.upper()} with {activation_name}. Logs saved to: {run_dir}")
        
        print("\nAll training runs completed!")
        
    finally:
        # Clean up TensorBoard process
        if tb_process:
            tb_process.terminate()
            print("\nTensorBoard stopped") 