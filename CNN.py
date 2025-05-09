import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datetime import datetime
import os

from helpers import get_device, make_writer, run_experiment

# — Hyperparameters —
batch_size = 64
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

# — Model Definition —
class SimpleCNN(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # grayscale input,32 feature maps, 3x3 kernel, keep dimension same
            activation(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # take in the 32 feature map, stacked as input, produce 64 feature maps... 
            activation(),
            nn.MaxPool2d(2), # turn each of the 64 horizontally "stacked" feature maps and turns their dimensions from 28x28 to 14x14?
            # ^^ reduce spatial dimensions so that network can learn more complex patterns (more variance due to less spatial info?)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # take in the 64 feature maps, produce 128 feature maps
            activation(),
            nn.MaxPool2d(2), ## our 128 feature maps go from 14x14 to 7x7
            nn.Flatten(), # above turns into a 1D vector [128 * 7 * 7]
            nn.Linear(128 * 7 * 7, 128), # using a classic MLP layer  
            activation(), 
            nn.Linear(128, 10) # output layer
        )
    def forward(self, x):
        return self.net(x)

model = SimpleCNN(activation)

if __name__ == "__main__":
    try:
        print("▶ Script start")
        device = get_device(model)

        print(f"▶ Beginning training for {epochs} epochs…")
        # Create a unique run directory with model and activation name
        model_name = "CNN"
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
