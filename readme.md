# USRA Survey - Modal GPU Training

This project uses Modal to run deep learning training on an RTX 5090 GPU while maintaining CUDA event recording and TensorBoard logging.

## Setup

1. Install Modal:

```bash
pip install modal
```

2. Login to Modal:

```bash
modal token new
```

3. Install other dependencies:

```bash
pip install -r requirements.txt
```

## Running the Training

To run the training on Modal's RTX 5090 GPU:

```bash
python modal_app.py
```

The training will run on Modal's infrastructure, and you'll see the progress in your terminal. The TensorBoard logs will be saved in the Modal container and can be accessed through the returned run directory.

## Features

- Runs on Modal's RTX 5090 GPU
- Maintains CUDA event recording for precise timing
- TensorBoard logging for visualization
- Automatic model checkpointing
- Learning rate scheduling
- Data augmentation

## Monitoring

The training progress can be monitored through:

1. Terminal output showing loss and accuracy metrics
2. TensorBoard logs (saved in the Modal container)
3. CUDA event timings for performance analysis

## Notes

- The code assumes the ImageNet dataset is available in the Modal container at `/root/data/ImageNet`
- Training logs and model checkpoints are saved in `/root/runs`
- The code uses ResNet-18 architecture by default
- Batch size is set to 256 for optimal GPU utilization
- Training runs for 90 epochs with cosine learning rate scheduling

installing cuda pytorch:

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python 3.11.12

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

run tensorboard: tensorboard --logdir=runs

modal volume get tensorboard-logs / ./runs
