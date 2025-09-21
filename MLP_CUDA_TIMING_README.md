# MLP CUDA Activation Timing

This enhanced version of MLP.py includes explicit CUDA event recording for timing each activation function and is wrapped with Modal decorators for cloud H100 execution.

## Features

### 🚀 CUDA Event Recording

- **Explicit timing**: Each activation layer is individually timed using `torch.cuda.Event`
- **High precision**: GPU-native timing with microsecond precision
- **Per-batch measurements**: Timing recorded for every forward pass
- **Automatic aggregation**: Average timing calculated per epoch

### ☁️ Modal Cloud Integration

- **H100 GPU**: Runs on NVIDIA H100 for maximum performance
- **Persistent storage**: Results saved to Modal volumes
- **Multiple activations**: Tests 8 different activation functions automatically
- **Tensorboard integration**: Real-time monitoring of training and timing

### 📊 Comprehensive Logging

- **CSV files**: Training metrics and activation timing saved as CSV for easy analysis
- **JSON stats**: Final results and configuration saved as JSON
- **Per-layer breakdown**: Individual timing for each activation layer
- **Efficiency metrics**: Accuracy vs timing analysis

## Quick Start

### 1. Install Dependencies

```bash
pip install modal torch torchvision matplotlib seaborn pandas
```

### 2. Setup Modal

```bash
modal token new
```

### 3. Run Training

```bash
# Run all activation functions on H100
modal run MLP.py

# Or use the helper script
python run_mlp_modal.py
```

### 4. Analyze Results

```bash
# After downloading results from Modal
python analyze_csv_results.py csv_logs/
```

## Architecture Details

### SimpleMLP Class Enhancements

```python
class SimpleMLP(nn.Module):
    def __init__(self, activation_fn):
        # ... standard layers ...

        # Store activation timings for analysis
        self.activation_times = {
            'layer_1': [],
            'layer_2': [],
            'layer_3': [],
            'layer_4': []
        }

    def forward(self, x):
        # ... linear operations ...

        # Time activation with CUDA events
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            x = self.activation1(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            self.activation_times['layer_1'].append(elapsed)
        else:
            x = self.activation1(x)
```

### Timing Methodology

1. **CUDA Events**: Use `torch.cuda.Event(enable_timing=True)` for GPU-native timing
2. **Synchronization**: `torch.cuda.synchronize()` ensures accurate measurements
3. **Per-batch Recording**: Each forward pass records timing data
4. **Epoch Aggregation**: Average timing calculated per epoch for TensorBoard

### Modal Cloud Setup

```python
# H100 GPU with persistent storage
@app.function(gpu="H100", volumes={"/root/tensorboard_logs": volume}, timeout=3600)
def train_mlp_with_timing(activation_fn, activation_name):
    # Training code with CUDA timing
```

## Tested Activation Functions

The system automatically tests these activation functions:

1. **ReLU** - Rectified Linear Unit
2. **GELU** - Gaussian Error Linear Unit
3. **Sigmoid** - Logistic function
4. **Tanh** - Hyperbolic tangent
5. **ELU** - Exponential Linear Unit
6. **LeakyReLU** - Leaky Rectified Linear Unit
7. **SiLU** - Sigmoid Linear Unit (Swish)
8. **Mish** - Self-regularized non-monotonic activation

## Output Files

Each training run produces:

### CSV Files

**training_metrics.csv** - Training progress over epochs

```csv
epoch,train_loss,train_acc,val_loss,val_acc
1,0.234,0.923,0.189,0.943
2,0.156,0.953,0.142,0.958
...
```

**activation_timings.csv** - Per-layer activation timing over epochs

```csv
epoch,layer_1,layer_2,layer_3,layer_4,total_time
1,0.025,0.021,0.018,0.014,0.078
2,0.024,0.020,0.017,0.013,0.074
...
```

### JSON Results

**final_results.json** - Final training statistics and configuration

```json
{
  "activation": "ReLU",
  "final_test_acc": 0.97,
  "final_train_acc": 0.98,
  "activation_times": {
    "layer_1": 0.023,
    "layer_2": 0.019,
    "layer_3": 0.015,
    "layer_4": 0.012
  },
  "total_activation_time": 0.069,
  "hyperparameters": {
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 0.001
  }
}
```

## Analysis Tools

### Visualization Scripts

- `analyze_csv_results.py` - Comprehensive CSV analysis with plots
- Generates timing comparisons, accuracy vs speed plots, heatmaps
- Creates markdown report with recommendations
- Outputs summary CSV for further analysis

### Key Metrics

- **Speed**: Total activation time per forward pass
- **Accuracy**: Final test set performance
- **Efficiency**: Accuracy per millisecond ratio
- **Layer Distribution**: Per-layer timing breakdown

## Performance Expectations

On H100 GPU with batch size 64:

- **Fastest activations**: ReLU, LeakyReLU (~0.02-0.05ms)
- **Medium activations**: GELU, SiLU (~0.05-0.08ms)
- **Slower activations**: Sigmoid, Tanh (~0.08-0.12ms)
- **Complex activations**: Mish (~0.10-0.15ms)

_Note: Actual timing depends on tensor sizes, GPU utilization, and other factors_

## Troubleshooting

### Common Issues

1. **Modal authentication**: Run `modal token new`
2. **GPU not available**: Ensure H100 is requested in Modal function
3. **Import errors**: Check that all files are included in Modal image
4. **Timing inconsistencies**: CUDA timing can vary with GPU load

### Debug Mode

Run locally for testing:

```bash
python MLP.py
```

This runs a single forward pass test without training.

## Integration with Existing Code

The enhanced MLP maintains compatibility with:

- ✅ `helpers.py` functions (get_device)
- ✅ Original training pipeline structure
- ✅ CSV logging for easy analysis
- ✅ Modal cloud deployment

## Next Steps

1. **Download results** from Modal volumes
2. **Run analysis** with `analyze_csv_results.py csv_logs/`
3. **View generated plots** and analysis report
4. **Compare activations** based on speed vs accuracy trade-offs

For questions or issues, check the CSV files and JSON results for detailed timing information.
