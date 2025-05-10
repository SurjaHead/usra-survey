import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datetime import datetime
import os
import shutil

from helpers import get_device, make_writer, run_experiment

# — Hyperparameters —
batch_size = 64
epochs = 10
learning_rate = 0.001
d_model = 32
nhead = 2
num_layers = 2
dim_feedforward = 64

# — Data Loaders —
full_train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1000, shuffle=False
)
test_loader = DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=1000, shuffle=False
)

# — Activation Function —
# Uncomment ONE of the following lines to choose the activation function:
# activation = nn.ReLU; activation_name = "ReLU"
# activation = nn.GELU; activation_name = "GELU"
# activation = nn.Sigmoid; activation_name = "Sigmoid"
# activation = nn.ELU; activation_name = "ELU"
# activation = nn.LeakyReLU; activation_name = "LeakyReLU"
activation = nn.Tanh; activation_name = "Tanh"

class SimpleTransformer(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Project input to d_model dimensions
        self.input_proj = nn.Linear(784, d_model)
        
        # Positional encoding (simple learned positional embeddings)
        self.pos_encoder = nn.Parameter(torch.randn(1, 784, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation=activation_fn(),
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 10)

    def forward(self, x):
        # Flatten and project input
        x = self.flatten(x)  # [batch_size, 784]
        x = x.unsqueeze(-1)  # [batch_size, 784, 1]
        x = x.expand(-1, -1, d_model)  # [batch_size, 784, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Global average pooling and project to output
        x = x.mean(dim=1)  # [batch_size, d_model]
        x = self.output_proj(x)  # [batch_size, 10]
        
        return x

if __name__ == "__main__":
    try:
        print("▶ Script start")
        model = SimpleTransformer(activation)
        device = get_device(model)

        print(f"▶ Beginning training for {epochs} epochs…")
        # Create a unique run directory with model and activation name
        model_name = "Transformer"
        base_run_dir = 'runs'
        model_run_prefix = f'{model_name}_{activation_name}'
        
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

        run_experiment(
            model,
            train_loader,
            val_loader,
            test_loader,
            device,
            writer,
            epochs=epochs,
            lr=learning_rate,
            is_gnn=False
        )
        print("✔ Training complete")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise 