# MLP Training Structure with Warmup

## Overview

- **Total Epochs**: 60
- **Warmup Epochs**: 10 (epochs 1-10)
- **Timing Epochs**: 50 (epochs 11-60)

## Training Phases

### Phase 1: Warmup (Epochs 1-10)

- **Purpose**: GPU warmup and model stabilization
- **Timing**: Disabled (no CUDA event recording)
- **Logging**: Only training metrics logged to `training_metrics.csv`
- **Output**: Shows remaining warmup epochs

### Phase 2: Timing (Epochs 11-60)

- **Purpose**: Accurate activation timing measurements
- **Timing**: Enabled (full CUDA event recording)
- **Logging**: Both training metrics and activation timing logged
- **CSV Timing Epochs**: Numbered 0-49 (50 timing epochs)

## File Outputs

### training_metrics.csv

- **Epochs**: 1-60 (all epochs)
- **Columns**: epoch, train_loss, train_acc, val_loss, val_acc

### activation_timings.csv

- **Epochs**: 0-49 (timing epochs only)
- **Columns**: epoch, layer_1, layer_2, layer_3, layer_4, total_time
- **Note**: Epoch 0 corresponds to actual training epoch 11

### final_results.json

```json
{
  "hyperparameters": {
    "total_epochs": 60,
    "warmup_epochs": 10,
    "timing_epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001
  }
}
```

## Console Output Examples

### Warmup Phase

```
Epoch  5 | Train 85.23% (0.4567) | Val 87.45% (0.3890) | 🔥 Warmup (6 epochs remaining)
```

### Timing Phase

```
Epoch 15 (Timing  4) | Train 92.34% (0.2345) | Val 91.78% (0.2567) | Total Activation Time: 0.089 ms
  layer_1: 0.022 ms
  layer_2: 0.021 ms
  layer_3: 0.023 ms
  layer_4: 0.023 ms
```

## Benefits

1. **Stable Timing**: Eliminates cold-start GPU variations
2. **Clean Data**: 50 epochs of consistent timing measurements
3. **Better Comparisons**: All activations measured under same conditions
4. **Clear Separation**: Training vs timing phases clearly distinguished
