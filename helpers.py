import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.data import DataLoader

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

def train_one_epoch(model, data_or_loader, criterion, optimizer, device, writer=None, epoch=None, is_gnn=False):
    """
    Runs a full training pass. Works for both MLP and GNN models.
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
                module._start_event = torch.cuda.Event(enable_timing=True)
                module._end_event = torch.cuda.Event(enable_timing=True)
                module._start_event.record()
            else:
                module._start_time = time.time()

        def post_hook(module, inp, outp):
            if device.type == "cuda":
                module._end_event.record()
                torch.cuda.synchronize()
                elapsed = module._start_event.elapsed_time(module._end_event)  # ms
            else:
                elapsed = (time.time() - module._start_time) * 1000.0  # ms
            
            # Get layer index
            layer_idx = 0
            for m in model.modules():
                if isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU, nn.GELU)):
                    if m == module:
                        break
                    layer_idx += 1
            name = f"Layer_{layer_idx}"
            activation_timings.setdefault(name, []).append(elapsed)

        # Register hooks on activation layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU, nn.GELU)):
                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(post_hook))

    try:
        if is_gnn:
            # GNN training with batched data
            for batch in data_or_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = out.argmax(dim=1)
                acc = (pred == batch.y).float().mean()
                total_loss += loss.item()
                correct += acc.item()
            # Average over batches
            total_loss /= len(data_or_loader)
            correct /= len(data_or_loader)
        else:
            # MLP training
            for X, y in data_or_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X.size(0)
                correct += (logits.argmax(1) == y).sum().item()

    finally:
        # Clean up hooks if we were measuring
        if writer is not None:
            for h in handles:
                h.remove()
            
            # Log activation timings to TensorBoard with descriptive names
            for name, times in activation_timings.items():
                if times:  # Only log if we have measurements
                    avg_time = sum(times) / len(times)
                    writer.add_scalar(
                        f"Activation_Time/{name}",
                        avg_time,
                        epoch
                    )
                    print(f"  {name}: {avg_time:.3f} ms (avg over {len(times)} calls)")

    if is_gnn:
        return total_loss, correct
    else:
        avg_loss = total_loss / len(data_or_loader.dataset)
        acc = correct / len(data_or_loader.dataset)
        return avg_loss, acc

def eval_on(model, data_or_loader, criterion, device, is_gnn=False):
    """Runs inference (no grad) over data. Works for both MLP and GNN models."""
    model.eval()
    with torch.no_grad():
        if is_gnn:
            # GNN evaluation with batched data
            total_loss, correct = 0.0, 0
            for batch in data_or_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                pred = out.argmax(dim=1)
                acc = (pred == batch.y).float().mean()
                total_loss += loss.item()
                correct += acc.item()
            # Average over batches
            total_loss /= len(data_or_loader)
            correct /= len(data_or_loader)
            return total_loss, correct
        else:
            # MLP evaluation
            total_loss, correct = 0.0, 0
            for X, y in data_or_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                total_loss += criterion(logits, y).item() * X.size(0)
                correct += (logits.argmax(1) == y).sum().item()
            avg_loss = total_loss / len(data_or_loader.dataset)
            acc = correct / len(data_or_loader.dataset)
            return avg_loss, acc

def run_experiment(model, train_data, val_data, test_data, device, writer, epochs=5, lr=1e-3, is_gnn=False):
    """Sets up Adam + CrossEntropyLoss, trains for epochs, logs metrics, and prints progress."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        t_loss, t_acc = train_one_epoch(model, train_data, criterion, optimizer, device, writer, epoch, is_gnn)
        v_loss, v_acc = eval_on(model, val_data, criterion, device, is_gnn)
        # Do NOT evaluate on test set here
        log_metrics(writer, epoch, t_loss, t_acc, v_loss, v_acc, None, None)
        print(f"Epoch {epoch:2d} | "
              f"Train {t_acc*100:5.2f}% ({t_loss:.4f}) | "
              f"Val {v_acc*100:5.2f}% ({v_loss:.4f})")

    # After all epochs, evaluate on test set ONCE
    e_loss, e_acc = eval_on(model, test_data, criterion, device, is_gnn)
    print(f"Final Test | Test {e_acc*100:5.2f}% ({e_loss:.4f})")
    summary_str = f"Final Test | Test {e_acc*100:5.2f}% ({e_loss:.4f})"
    writer.add_text("Final Test Summary", summary_str, epochs)
    writer.close()
