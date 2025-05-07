import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helpers import (
    get_device,
    make_writer,
    run_experiment,
    profile_batch,
    time_activation_layers    # ← new import
)

# — Hyperparameters —
batch_size = 64
epochs     = 5
learning_rate = 1e-3

# — Data Loaders —
train_loader = DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    datasets.MNIST('data', train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=1000, shuffle=False
)
test_loader = val_loader  # reuse for simplicity

# — Model Definition —
model = nn.Sequential(
    nn.Flatten(),         # 28×28 → 784
    nn.Linear(784, 128),  # hidden layer
    nn.ReLU(),            # activation
    nn.Linear(128, 10)    # output logits
)

if __name__ == "__main__":
    print("▶ Script start")
    device = get_device(model)

    print("▶ Running profiler on 1 batch…")
    profile_batch(model, train_loader, device, steps=1)
    print("✔ Profiler done")


    print("▶ Timing activations over 3 batches…")
    time_activation_layers(model, train_loader, device, num_batches=3)
    print("✔ Activation timing done\n")

    print(f"▶ Beginning training for {epochs} epochs…")
    writer = make_writer("runs/exp1")
    run_experiment(
        model,
        train_loader, val_loader, test_loader,
        device, writer,
        epochs=epochs, lr=learning_rate
    )
    print("✔ Training complete")
