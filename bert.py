from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset
import os
import evaluate
import numpy as np
import torch
from transformers.models.bert.modeling_bert import BertIntermediate
from transformers.activations import GELUActivation, QuickGELUActivation  # Add QuickGELU import
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
import time

@dataclass
class IntermediateConfig:
    hidden_size: int
    intermediate_size: int
    hidden_act: str

# Create a custom BertIntermediate class with timing
class TimedBertIntermediate(BertIntermediate):
    def __init__(self, config, original_intermediate):
        # Create a minimal config with just what we need
        intermediate_config = IntermediateConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )
        super().__init__(intermediate_config)
        # Copy the weights from original intermediate
        self.dense.weight = original_intermediate.dense.weight
        self.dense.bias = original_intermediate.dense.bias
        self.intermediate_act_fn = original_intermediate.intermediate_act_fn
        
        # Timing statistics
        self.total_time = 0
        self.num_calls = 0
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Run the dense layer
        hidden_states = self.dense(hidden_states)
        
        # Time the activation function
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            hidden_states = self.intermediate_act_fn(hidden_states)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)  # ms
        else:
            start_time = time.time()
            hidden_states = self.intermediate_act_fn(hidden_states)
            elapsed = (time.time() - start_time) * 1000.0  # ms
            
        self.total_time += elapsed
        self.num_calls += 1
        
        return hidden_states

# Create data directory
os.makedirs("data-BERT", exist_ok=True)
cache_dir = os.path.join("data-BERT", "cache")

# Load model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", cache_dir=cache_dir)

# Create model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, cache_dir=cache_dir)

# Replace all intermediate layers with our timed version
for layer in model.bert.encoder.layer:
    original_intermediate = layer.intermediate
    layer.intermediate = TimedBertIntermediate(model.config, original_intermediate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
model.to(device)

# Function to print timing stats
def print_timing_stats():
    print("\nActivation Timing Statistics:")
    total_avg = 0
    for i, layer in enumerate(model.bert.encoder.layer):
        avg_time = layer.intermediate.total_time / layer.intermediate.num_calls if layer.intermediate.num_calls > 0 else 0
        total_avg += avg_time
        print(f"Layer {i}: {avg_time:.6f} ms average ({layer.intermediate.num_calls} calls)")
    print(f"Average across all layers: {(total_avg/12):.6f} ms")

# Function to switch activation functions
def switch_activation(activation_type="relu"):
    print(f"\nSwitching to {activation_type} activation")
    model.config.hidden_act = activation_type
    for layer in model.bert.encoder.layer:
        if activation_type == "relu":
            layer.intermediate.intermediate_act_fn = torch.nn.ReLU()
        elif activation_type == "sigmoid":
            layer.intermediate.intermediate_act_fn = torch.nn.Sigmoid()
        elif activation_type == "elu":
            layer.intermediate.intermediate_act_fn = torch.nn.ELU()
        elif activation_type == "gelu":
            layer.intermediate.intermediate_act_fn = GELUActivation()
        elif activation_type == "quick_gelu":  # Add QuickGELU option
            layer.intermediate.intermediate_act_fn = QuickGELUActivation()
        elif activation_type == "tanh":
            layer.intermediate.intermediate_act_fn = torch.nn.Tanh()
        elif activation_type == "leaky_relu":
            layer.intermediate.intermediate_act_fn = torch.nn.LeakyReLU(0.1)
        elif activation_type == "silu" or activation_type == "swish":
            layer.intermediate.intermediate_act_fn = torch.nn.SiLU()
        elif activation_type == "mish":
            class Mish(torch.nn.Module):
                def forward(self, x):
                    return x * torch.tanh(torch.nn.functional.softplus(x))
            layer.intermediate.intermediate_act_fn = Mish()
    verify_activation_swapped()

def verify_activation_swapped():
    print("\nVerifying activations in each layer:")
    for i, layer in enumerate(model.bert.encoder.layer):
        fn = layer.intermediate.intermediate_act_fn
        print(f" Layer {i:2d}: {fn.__class__.__name__}")


# Load SST-2 dataset
dataset = load_dataset("glue", "sst2", cache_dir=cache_dir)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

# Tokenize datasets
tokenized_train = dataset["train"].map(
    tokenize_function, 
    batched=True,
    remove_columns=["sentence", "idx"]  # Only remove these columns, keep 'label'
)
tokenized_val = dataset["validation"].map(
    tokenize_function, 
    batched=True,
    remove_columns=["sentence", "idx"]  # Only remove these columns, keep 'label'
)

# Rename 'label' to 'labels' to match what the model expects
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_val = tokenized_val.rename_column("label", "labels")

# Load GLUE metrics
metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

switch_activation("quick_gelu")  # Try QuickGELU activation

training_args = TrainingArguments(
    output_dir=os.path.join("data-BERT", "runs", f"BERT_{model.config.hidden_act}"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-5,  # Increased from 2e-5 to 5e-5
    logging_steps=100,
    save_steps=500,
    save_total_limit=1,
    remove_unused_columns=False,
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.01,
    max_grad_norm=1.0,  # Keep gradient clipping
    eval_steps=100,
    eval_strategy="steps",
    report_to=["tensorboard"],  # Add TensorBoard reporting
    logging_dir=os.path.join("data-BERT", "runs", f"BERT_{model.config.hidden_act}")  # Explicitly set logging directory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Create a global SummaryWriter for activation timings in the same directory as training logs
activation_writer = SummaryWriter(os.path.join("data-BERT", "runs", f"BERT_{model.config.hidden_act}"))

# Custom TrainerCallback to log activation timings at every step
class ActivationTimingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        global_step = state.global_step
        total_time = 0  # Track total time across all layers
        
        for i, layer in enumerate(model.bert.encoder.layer):
            avg_time = layer.intermediate.total_time / layer.intermediate.num_calls if layer.intermediate.num_calls > 0 else 0
            activation_writer.add_scalar(f"activation_time/layer_{i}", avg_time, global_step)
            total_time += avg_time  # Add this layer's time to total
        
        # Log the total activation time across all layers
        activation_writer.add_scalar("activation_time/total", total_time, global_step)

    def on_train_end(self, args, state, control, **kwargs):
        activation_writer.close()

# Add the callback to the trainer
trainer.add_callback(ActivationTimingCallback())

# Train and evaluate
print("Starting training on SST-2...")

# Add warm-up period
print("Running warm-up period...")
warmup_steps = 100
for _ in range(warmup_steps):
    batch = next(iter(trainer.get_train_dataloader()))
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        _ = model(**batch)

# Reset timing counters after warm-up
for layer in model.bert.encoder.layer:
    layer.intermediate.total_time = 0
    layer.intermediate.num_calls = 0

print("Warm-up complete. Starting actual training...")
train_results = trainer.train()
eval_results = trainer.evaluate()

# Print activation timing stats after training
print_timing_stats()

print("\nSST-2 Results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Training Loss: {train_results.training_loss:.4f}")
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")

# Save the model
model.save_pretrained(os.path.join("data-BERT", "sst2-finetuned"))
print("\nModel saved to data-BERT/sst2-finetuned")