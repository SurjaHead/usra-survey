import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datetime import datetime
import os

from helpers import (
    get_device,
    make_writer,
    run_experiment,
)

# — Hyperparameters —
batch_size = 64 # lets try and see if increasing batch size will improve performance
epochs     = 10
learning_rate = 0.001

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
activation = nn.Sigmoid; activation_name = "Sigmoid"
# activation = nn.Tanh; activation_name = "Tanh"

model = nn.Sequential(
    nn.Flatten(),         # 28×28 → 784
    nn.Linear(784, 512),  # layer 1
    activation(),         # activation 1
    nn.Linear(512, 256),  # layer 2
    activation(),         # activation 2
    nn.Linear(256, 128),  # layer 3
    activation(),         # activation 3
    nn.Linear(128, 64),   # layer 4
    activation(),         # activation 4
    nn.Linear(64, 10)     # output layer
)

if __name__ == "__main__":
    try:
        print("▶ Script start")
        device = get_device(model)

        print(f"▶ Beginning training for {epochs} epochs…")
        # Create a unique run directory with model and activation name
        model_name = "MLP"
        run_dir = f"runs/{model_name}_{activation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = make_writer(run_dir)
        run_experiment(
            model,
            train_loader, val_loader, test_loader,
            device, writer,
            epochs=epochs, lr=learning_rate
        )
        print("✔ Training complete")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
