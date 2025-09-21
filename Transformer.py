import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datetime import datetime
import os
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
        "torchvision", 
        "transformers",
        "tensorboard"  # Add tensorboard for helpers.py compatibility
    ])  # Then install additional packages
    .add_local_python_source("helpers")
)

volume = modal.Volume.from_name("bert-csv-logs", create_if_missing=True)
app = modal.App("bert-timing", image=image)

# — Hyperparameters (matching BERT specification) —
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 3
TRAIN_VAL_SPLIT = 0.9  # 90/10 split
DIMENSIONALITY = 768  # BERT base dimensionality
NUM_HEADS = 12  # Multi-head attention heads
NUM_ENCODERS = 12  # 12x Encoders
FFN_HIDDEN = 3072  # FFN: 768 -> 3072 -> 768
DROPOUT = 0.0  # No dropout mentioned in image

# Activation functions for testing
ACTIVATION_FUNCTIONS = {
    # 'ReLU': nn.ReLU,
    # 'GELU': nn.GELU,
    # 'Sigmoid': nn.Sigmoid,
    # 'Tanh': nn.Tanh,
    # 'ELU': nn.ELU,
    'LeakyReLU': nn.LeakyReLU,
    'SiLU': nn.SiLU,
    'Mish': nn.Mish
}

class BERTEncoder(nn.Module):
    """Single BERT Encoder layer with CUDA timing for FFN activation"""
    def __init__(self, d_model, nhead, dim_feedforward, activation_fn, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=DROPOUT, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN: 768 -> 3072 -> 768 with activation timing
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation_fn()  # This is what we time
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # CUDA timing storage for this layer's activation
        self.activation_times = []
    
    def forward(self, src, src_mask=None):
        # Multi-head attention with residual connection
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + src2)
        
        # FFN with activation timing
        src2 = self.linear1(src)
        
        # Time the activation function
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        src2 = self.activation(src2)  # This is the timed operation
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        self.activation_times.append(elapsed_time)
        
        src2 = self.linear2(src2)
        src = self.norm2(src + src2)
        
        return src
    
    def get_activation_times(self):
        """Return activation times for this layer"""
        return self.activation_times.copy()
    
    def reset_activation_times(self):
        """Reset activation timing storage"""
        self.activation_times = []

class BERTModel(nn.Module):
    """BERT-like model matching the specification in the image"""
    def __init__(self, activation_fn, vocab_size=784, num_classes=10):
        super().__init__()
        
        # Input projection and tokenization
        # For MNIST: treat each pixel as a token, so 784 tokens per image
        self.token_embedding = nn.Linear(1, DIMENSIONALITY)  # Each pixel -> 768 dim
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, vocab_size + 1, DIMENSIONALITY))  # +1 for CLS token
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, DIMENSIONALITY))
        
        # 12x BERT Encoders
        self.encoders = nn.ModuleList([
            BERTEncoder(DIMENSIONALITY, NUM_HEADS, FFN_HIDDEN, activation_fn, i) 
            for i in range(NUM_ENCODERS)
        ])
        
        # CLS token pooling + classification head
        self.classifier = nn.Linear(DIMENSIONALITY, num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Tokenization: treat each pixel as a token
        x = x.view(batch_size, 784, 1)  # [batch_size, 784, 1]
        x = self.token_embedding(x)  # [batch_size, 784, 768]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, 768]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, 785, 768]
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Pass through 12x encoders
        for encoder in self.encoders:
            x = encoder(x)
        
        # CLS token pooling (take first token)
        cls_output = x[:, 0, :]  # [batch_size, 768]
        
        # Classification
        logits = self.classifier(cls_output)  # [batch_size, num_classes]
        
        return logits
    
    def get_all_activation_times(self):
        """Return all activation times for each encoder layer"""
        all_times = {}
        for i, encoder in enumerate(self.encoders):
            all_times[f'encoder_{i}_activation'] = encoder.get_activation_times()
        return all_times
    
    def get_average_activation_times(self):
        """Return average activation time for each encoder layer"""
        avg_times = {}
        for i, encoder in enumerate(self.encoders):
            times = encoder.get_activation_times()
            if times:
                avg_times[f'encoder_{i}_activation'] = sum(times) / len(times)
            else:
                avg_times[f'encoder_{i}_activation'] = 0.0
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
        """Reset activation timing storage for all encoders"""
        for encoder in self.encoders:
            encoder.reset_activation_times()

@app.function(
    volumes={"/data": volume},
    gpu="H100",
    timeout=7200,  # 2 hours
    scaledown_window=300
)
def train_bert_with_timing(activation_name: str):
    """Train BERT with CUDA timing and CSV logging"""
    print(f"🚀 Training BERT with {activation_name} activation")
    print(f"📊 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    # Data preparation (90/10 train/val split as specified)
    print("📁 Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    full_train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_size = int(TRAIN_VAL_SPLIT * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📈 Dataset Info:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    # Get activation function
    activation_fn = ACTIVATION_FUNCTIONS[activation_name]
    
    # Create model
    model = BERTModel(activation_fn)
    device = get_device(model)
    model = model.to(device)
    
    print(f"🔧 Model on device: {device}")
    print(f"🏗️  Architecture: BERT with {NUM_ENCODERS} encoders, {NUM_HEADS} heads")
    print(f"🔧 FFN: {DIMENSIONALITY} → {FFN_HIDDEN} → {DIMENSIONALITY}")
    
    # Setup logging directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"BERT_{activation_name}_{timestamp}"
    
    # Local paths for this run
    local_run_dir = f"/tmp/{run_name}"
    os.makedirs(local_run_dir, exist_ok=True)
    
    # CSV file paths
    training_log_path = os.path.join(local_run_dir, "training_log.csv")
    activation_log_path = os.path.join(local_run_dir, "activation_timings.csv")
    
    # Create CSV headers
    with open(training_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'step', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    # Get activation layer names for CSV header (all 12 encoder activations)
    activation_layer_names = [f'encoder_{i}_activation' for i in range(NUM_ENCODERS)]
    
    with open(activation_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch', 'step'] + activation_layer_names + ['total_time']
        writer.writerow(header)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🎯 Starting training for {EPOCHS} epochs...")
    
    global_step = 0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Reset activation timings for this step
            model.reset_activation_times()
            
            optimizer.zero_grad()
            output = model(data)  # This will record activation timings
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate step metrics
            pred = output.argmax(dim=1, keepdim=True)
            step_correct = pred.eq(target.view_as(pred)).sum().item()
            step_total = target.size(0)
            step_acc = step_correct / step_total
            
            # Get activation timings for this step
            total_activation_time = model.get_average_total_activation_time_per_forward_pass()
            avg_activation_times = model.get_average_activation_times()
            
            # Log step data to CSV
            with open(activation_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                timing_row = [epoch, global_step]
                for layer_name in activation_layer_names:
                    timing_row.append(avg_activation_times.get(layer_name, 0.0))
                timing_row.append(total_activation_time)
                writer.writerow(timing_row)
            
            # Accumulate epoch metrics
            epoch_train_loss += loss.item()
            epoch_train_correct += step_correct
            epoch_train_total += step_total
            
            global_step += 1
        
        # Calculate epoch averages
        train_loss = epoch_train_loss / len(train_loader)
        train_acc = epoch_train_correct / epoch_train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)  # This will also record activation timings but won't be logged per step
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Log epoch summary to training log
        with open(training_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, global_step-1, train_loss, train_acc, val_loss, val_acc])
        
        print(f"Epoch {epoch+1:2d} | "
              f"Train {train_acc*100:5.2f}% ({train_loss:.4f}) | "
              f"Val {val_acc*100:5.2f}% ({val_loss:.4f}) | "
              f"Steps completed: {global_step}")
    
    # Final test (use validation set as test since MNIST test isn't needed for timing)
    print(f"\n🎯 Final Validation Accuracy: {val_acc:.4f}")
    
    # Save final results
    final_results = {
        'model': 'BERT',
        'dataset': 'MNIST',
        'activation': activation_name,
        'final_train_accuracy': float(train_acc),
        'final_val_accuracy': float(val_acc),
        'final_train_loss': float(train_loss),
        'final_val_loss': float(val_loss),
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'num_encoders': NUM_ENCODERS,
        'num_heads': NUM_HEADS,
        'dimensionality': DIMENSIONALITY,
        'ffn_hidden': FFN_HIDDEN,
        'architecture': f"BERT: {NUM_ENCODERS} encoders, {NUM_HEADS} heads, FFN {DIMENSIONALITY}→{FFN_HIDDEN}→{DIMENSIONALITY}",
        'total_steps': global_step,
        'timestamp': timestamp
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
        'val_accuracy': val_acc,
        'total_steps': global_step
    }

@app.local_entrypoint()
def main():
    """Run BERT training for all activation functions"""
    print("🚀 Starting BERT activation function comparison on Modal")
    
    results = []
    for activation_name in ACTIVATION_FUNCTIONS.keys():
        print(f"\n{'='*60}")
        print(f"🧠 Training with {activation_name} activation")
        print('='*60)
        
        result = train_bert_with_timing.remote(activation_name)
        results.append(result)
        
        print(f"✅ Completed {activation_name}: Val Acc = {result['val_accuracy']:.4f}, "
              f"Total Steps = {result['total_steps']}")
    
    print(f"\n🎉 All training completed!")
    print("\n📊 Summary:")
    for result in results:
        print(f"  {result['run_name']}: {result['val_accuracy']:.4f} acc, {result['total_steps']} steps")

# For local testing
if __name__ == "__main__":
    # Test locally with GELU activation (as specified in image)
    print("🧪 Local testing with GELU activation")
    activation_name = "GELU"
    activation_fn = ACTIVATION_FUNCTIONS[activation_name]
    
    # Create small model for testing
    model = BERTModel(activation_fn)
    device = get_device(model)
    model = model.to(device)
    
    print(f"✅ Model created successfully!")
    print(f"📊 Architecture: BERT with {NUM_ENCODERS} encoders, {NUM_HEADS} heads")
    print(f"🔧 FFN: {DIMENSIONALITY} → {FFN_HIDDEN} → {DIMENSIONALITY}")
    print(f"🎮 Device: {device}")
    
    # Test forward pass with dummy data
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"🔍 Output shape: {output.shape}")
        
        # Check activation timing
        total_time = model.get_average_total_activation_time_per_forward_pass()
        avg_times = model.get_average_activation_times()
        print(f"⏱️  Total activation time: {total_time:.3f}ms")
        print(f"🔧 Individual encoder timings:")
        for layer, time in avg_times.items():
            print(f"   {layer}: {time:.3f}ms")