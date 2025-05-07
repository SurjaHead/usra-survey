import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import (
    profile, record_function, ProfilerActivity,
    tensorboard_trace_handler
)

import time

def get_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    if not torch.cuda.is_available():
        print("WARNING: No CUDA device found; continuing on CPU.")
    return device


def make_writer(logdir="runs/exp1"):
    """
    Returns a TensorBoard SummaryWriter pointed at `logdir`.
    """
    return SummaryWriter(logdir)

def log_metrics(writer, epoch,
                train_loss, train_acc,
                val_loss,   val_acc,
                test_loss,  test_acc):
    """
    Logs scalar metrics to TensorBoard under clear tags.
    """
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Acc/train",  train_acc,  epoch)
    writer.add_scalar("Loss/val",   val_loss,   epoch)
    writer.add_scalar("Acc/val",    val_acc,    epoch)
    writer.add_scalar("Loss/test",  test_loss,  epoch)
    writer.add_scalar("Acc/test",   test_acc,   epoch)

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs a full training pass over `loader`.
    Returns: (average loss, accuracy)
    """
    model.train()
    total_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct    += (logits.argmax(1) == y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc      = correct / len(loader.dataset)
    return avg_loss, acc

def eval_on(model, loader, criterion, device):
    """
    Runs inference (no grad) over `loader`.
    Returns: (average loss, accuracy)
    """
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * X.size(0)
            correct    += (logits.argmax(1) == y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    acc      = correct / len(loader.dataset)
    return avg_loss, acc

def run_experiment(model,
                   train_loader, val_loader, test_loader,
                   device, writer,
                   epochs=5, lr=1e-3):
    """
    • Sets up Adam + CrossEntropyLoss.
    • For each epoch: trains, validates, tests.
    • Logs metrics via `log_metrics` and prints progress.
    • Closes the writer at end.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = eval_on(model, val_loader,   criterion,        device)
        e_loss, e_acc = eval_on(model, test_loader,  criterion,        device)

        log_metrics(writer, epoch, t_loss, t_acc, v_loss, v_acc, e_loss, e_acc)
        print(f"Epoch {epoch:2d} | "
              f"Train {t_acc*100:5.2f}% ({t_loss:.4f}) | "
              f"Val {v_acc*100:5.2f}% ({v_loss:.4f}) | "
              f"Test {e_acc*100:5.2f}% ({e_loss:.4f})")

    writer.close()

def profile_batch(model, loader, device,
                  log_dir="runs/profile", steps=5):
    """
    Profiles `steps` mini-batches from `loader`:
    • Captures CPU & CUDA op timings, shapes, memory, stacks.
    • Emits trace files under `log_dir` for TensorBoard's Profile tab.
    """
    model.eval()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler(log_dir)
    ) as prof:
        for i, (X, _) in enumerate(loader):
            X = X.to(device)
            with record_function("model_inference"):
                _ = model(X)
            prof.step()
            if i+1 >= steps:
                break
    print("  → [profile_batch] finished profiling")

# the num_batches argument is used to for warmup and averaging
def time_activation_layers(model, loader, device, num_batches=1):
    """
    Measures wall‐clock time of each activation module (ReLU, Sigmoid, etc.)
    on `num_batches` mini‐batches from `loader`. Prints the average ms per layer.
    """
    # Dictionary to accumulate times per module
    timings = {}
    handles = []

    # Pre‐ and post‐forward hooks
    def pre_hook(module, inp):
        # Synchronize GPU if needed, then stamp start time
        if device.type == "cuda":
            torch.cuda.synchronize()
        module._start_time = time.time()

    def post_hook(module, inp, outp):
        # Synchronize again, compute elapsed
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - module._start_time) * 1000.0  # ms
        name = f"{module.__class__.__name__}_{id(module)}"
        timings.setdefault(name, []).append(elapsed)

    # Register hooks on every activation module in the network
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU)):
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))

    # Run a few batches (no grads) and collect timings
    model.eval()
    with torch.no_grad():
        for i, (X, _) in enumerate(loader):
            X = X.to(device)
            _ = model(X)
            if i + 1 >= num_batches:
                break

    # Remove hooks
    for h in handles:
        h.remove()

    # Print summary
    print("Activation timing (avg ms over", len(next(iter(timings.values()))), "calls):")
    for name, times in timings.items():
        avg = sum(times) / len(times)
        print(f"  {name}: {avg:.3f} ms")

    return timings