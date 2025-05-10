import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datetime import datetime
import os
import shutil

from helpers import get_device, make_writer, run_experiment

# — Hyperparameters —
batch_size = 64
epochs     = 50
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
# activation = nn.Sigmoid; activation_name = "Sigmoid"
# activation = nn.ELU; activation_name = "ELU"
# activation = nn.LeakyReLU; activation_name = "LeakyReLU"
activation = nn.Tanh; activation_name = "Tanh"

# — Model Definition —
class SimpleCNN(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        # grayscale input, 32 feature maps, 3x3 kernel, keep dimension same
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.activation1 = activation_fn()
        # take in the 32 feature map, stacked as input, produce 64 feature maps...
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.activation2 = activation_fn()
        # turn each of the 64 horizontally "stacked" feature maps and turns their dimensions from 28x28 to 14x14
        # reduce spatial dimensions so that network can learn more complex patterns (more variance due to less spatial info?)
        self.pool1 = nn.MaxPool2d(2)
        # take in the 64 feature maps, produce 128 feature maps
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.activation3 = activation_fn()
        # our 128 feature maps go from 14x14 to 7x7
        self.pool2 = nn.MaxPool2d(2)
        # above turns into a 1D vector [128 * 7 * 7]
        self.flatten = nn.Flatten()
        # using a classic MLP layer
        self.linear1 = nn.Linear(128 * 7 * 7, 128)
        self.activation4 = activation_fn()
        # output layer
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation4(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    try:
        print("▶ Script start")
        model = SimpleCNN(activation)
        device = get_device(model)

        print(f"▶ Beginning training for {epochs} epochs…")
        # Create a unique run directory with model and activation name
        model_name = "CNN"
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
