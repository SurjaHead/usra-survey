import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import time

def get_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    if not torch.cuda.is_available():
        print("WARNING: No CUDA device found; continuing on CPU.")
    return device

def make_writer(logdir="runs/exp1"):
    """Returns a TensorBoard SummaryWriter pointed at `logdir`."""
    return SummaryWriter(logdir)

def log_metrics(writer, epoch, train_loss, train_acc, val_loss, val_acc, test_loss=None, test_acc=None):
    """Logs scalar metrics to TensorBoard under clear tags."""
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Acc/train",  train_acc,  epoch)
    writer.add_scalar("Loss/val",   val_loss,   epoch)
    writer.add_scalar("Acc/val",    val_acc,    epoch)
    if test_loss is not None:
        writer.add_scalar("Loss/test",  test_loss,  epoch)
    if test_acc is not None:
        writer.add_scalar("Acc/test",   test_acc,   epoch)

def train_one_epoch(model, loader, criterion, optimizer, device, writer=None, epoch=None):
    """
    Runs a full training pass over `loader`. 
    If writer and epoch are provided, logs activation timings to TensorBoard.
    Returns: (average loss, accuracy)
    """
    model.train()
    total_loss, correct = 0.0, 0
    
    # Setup activation timing if writer is provided
    handles = []
    activation_timings = {}
    
    if writer is not None:
        def pre_hook(module, inp):
            if device.type == "cuda":
                torch.cuda.synchronize()
            module._start_time = time.time()

        def post_hook(module, inp, outp):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.time() - module._start_time) * 1000.0  # ms
            # Get layer index and input size for more detailed naming
            layer_idx = list(model).index(module)
            input_size = inp[0].size()
            name = f"Layer_{layer_idx}_ReLU_{input_size[1]}"
            activation_timings.setdefault(name, []).append(elapsed)

        # Register hooks on activation layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU, nn.GELU)):
                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(post_hook))

    try:
        for X, y in loader:                      # once for each batch. loop iterates once for each batch
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward() # backward pass 
            optimizer.step() # grad. desc. (weight update)

            total_loss += loss.item() * X.size(0)
            correct    += (logits.argmax(1) == y).sum().item()

    finally:
        # Clean up hooks if we were measuring
        if writer is not None:
            for h in handles:
                h.remove()
            
            # Log activation timings to TensorBoard with descriptive names
            for name, times in activation_timings.items():
                avg_time = sum(times) / len(times)
                writer.add_scalar(
                    f"Activation_Time/{name}",
                    avg_time,
                    epoch
                )
                print(f"  {name}: {avg_time:.3f} ms (avg over {len(times)} calls)")

    avg_loss = total_loss / len(loader.dataset)
    acc      = correct / len(loader.dataset)
    return avg_loss, acc

def eval_on(model, loader, criterion, device):
    """Runs inference (no grad) over `loader`. Returns: (average loss, accuracy)"""
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

def run_experiment(model, train_loader, val_loader, test_loader, device, writer, epochs=5, lr=1e-3):
    """Sets up Adam + CrossEntropyLoss, trains for epochs, logs metrics, and prints progress."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, epoch)
        v_loss, v_acc = eval_on(model, val_loader, criterion, device)
        # Do NOT evaluate on test set here
        log_metrics(writer, epoch, t_loss, t_acc, v_loss, v_acc, None, None)
        print(f"Epoch {epoch:2d} | "
              f"Train {t_acc*100:5.2f}% ({t_loss:.4f}) | "
              f"Val {v_acc*100:5.2f}% ({v_loss:.4f})")

    # After all epochs, evaluate on test set ONCE
    e_loss, e_acc = eval_on(model, test_loader, criterion, device)
    print(f"Final Test | Test {e_acc*100:5.2f}% ({e_loss:.4f})")
    summary_str = f"Final Test | Test {e_acc*100:5.2f}% ({e_loss:.4f})"
    writer.add_text("Final Test Summary", summary_str, epochs)
    writer.close()