# PyTorch Lightning Tuner - Complete Guide

## Overview

The `Tuner` class in PyTorch Lightning provides **automatic hyperparameter optimization** for two critical training parameters:

1. **Learning Rate** (`lr_find()`) - Find optimal LR using the LR Range Test
2. **Batch Size** (`scale_batch_size()`) - Find maximum batch size that fits in GPU memory

Both methods run **BEFORE** `trainer.fit()` and typically take only 1-3 minutes.

## The Two Tuning Methods

### 1. Learning Rate Finder (`lr_find()`)

**What it does:**
- Performs the "LR Range Test" (Leslie Smith, 2015)
- Trains the model for ~100-200 iterations
- Exponentially increases learning rate from very small (1e-8) to very large (1.0)
- Records the loss at each LR value
- Suggests the optimal LR based on the steepest descent in the loss curve

**Why it's important:**
- Learning rate is the **single most important hyperparameter** for neural network training
- Too low → slow convergence, gets stuck in local minima
- Too high → unstable training, diverges
- Optimal LR changes based on model architecture, dataset, and batch size

**How it works:**
```python
from lightning.pytorch.tuner import Tuner

trainer = Trainer(...)
tuner = Tuner(trainer)

# Run LR finder
lr_finder = tuner.lr_find(
    model,
    datamodule=datamodule,
    min_lr=1e-8,      # Start LR
    max_lr=1.0,       # End LR
    num_training=100, # Number of iterations
)

# Get suggested LR
suggested_lr = lr_finder.suggestion()
print(f"Suggested LR: {suggested_lr}")

# Plot the results
fig = lr_finder.plot(suggest=True)
fig.show()

# Update model and train
model.hparams.learning_rate = suggested_lr
trainer.fit(model, datamodule=datamodule)
```

**The LR finder looks for:**
- `model.lr` (if exists)
- `model.learning_rate` (if exists)
- `model.hparams.lr` (if exists)
- `model.hparams.learning_rate` (if exists)

It automatically updates the first one it finds.

### 2. Batch Size Finder (`scale_batch_size()`)

**What it does:**
- Finds the largest batch size that fits in GPU memory
- Two modes:
  - **Power mode** (default): Exponential search (2→4→8→16→32→64...)
  - **Binary search mode**: More precise but slower

**Why it's important:**
- Larger batch sizes → faster training (better GPU utilization)
- But: Too large → OOM (Out of Memory) error
- Optimal batch size depends on: model size, GPU memory, image resolution

**How it works:**
```python
tuner.scale_batch_size(
    model,
    datamodule=datamodule,
    mode="power",    # or "binsearch"
    init_val=2,      # Starting batch size
    max_trials=25,   # Max iterations
)

# Automatically updates datamodule.hparams.batch_size
print(f"Optimal batch size: {datamodule.hparams.batch_size}")
```

## Our Implementation

### CLI Usage

**Run both tuning methods:**
```bash
uv run traenslenzor/doc_classifier/run.py \
    --config_path .configs/alexnet_scratch.toml \
    --tune-learning-rate \
    --tune-batch-size
```

**Run only LR finder:**
```bash
uv run traenslenzor/doc_classifier/run.py \
    --config_path .configs/alexnet_scratch.toml \
    --tune-learning-rate
```

**Run only batch size finder:**
```bash
uv run traenslenzor/doc_classifier/run.py \
    --config_path .configs/alexnet_scratch.toml \
    --tune-batch-size
```

### What Happens

When you run with `--tune-learning-rate --tune-batch-size`:

1. **Batch Size Tuning** (runs first):
   ```
   [ExperimentConfig::tuner]: Running batch size tuning...
   [ExperimentConfig::tuner]: Mode: power, init_val: 2, max_trials: 25
   [ExperimentConfig::tuner]: ✓ Optimal batch size found: 128
   ```

2. **Learning Rate Finding** (runs second):
   ```
   [ExperimentConfig::tuner]: Running learning rate finder (LR range test)...
   [ExperimentConfig::tuner]: Training for ~100 iterations while exponentially increasing LR...
   [ExperimentConfig::tuner]: ✓ Suggested learning rate: 3.16e-03
   [ExperimentConfig::tuner]:   Updated optimizer LR: 1.00e-03 → 3.16e-03
   [ExperimentConfig::tuner]:   LR finder plot saved to: .logs/lr_finder_plot.png
   ```

3. **Normal Training** (with optimized hyperparameters):
   ```
   [ExperimentConfig::train]: Starting training (fit)...
   ```

### The LR Finder Plot

The saved plot (`.logs/lr_finder_plot.png`) shows:
- **X-axis**: Learning rate (log scale)
- **Y-axis**: Loss
- **Red dot**: Suggested optimal LR (steepest descent point)

**How to interpret:**
```
Loss
 |
 |     ╱╲  ← Too high: loss explodes
 |    ╱  ╲
 |   ╱    ╲
 |  ╱      ╲
 | ╱        ●  ← Optimal: steepest descent
 |╱          ╲
 |____________╲_____ → Learning Rate (log)
 ↑            ↑
 Too low     Sweet spot
```

## When to Use Tuner

### Initial Setup (Once per Model/GPU/Dataset)

1. **First run with tuning:**
   ```bash
   uv run traenslenzor/doc_classifier/run.py \
       --config_path .configs/alexnet_scratch.toml \
       --tune-batch-size \
       --tune-learning-rate
   ```

2. **Note the optimal values from logs:**
   ```
   Optimal batch size: 128
   Suggested learning rate: 3.16e-03
   ```

3. **Update your config:**
   ```toml
   [datamodule_config]
   batch_size = 128

   [module_config.optimizer]
   learning_rate = 0.00316

   [module_config.scheduler]
   max_lr = 0.01  # Adjust to ~3x the base LR
   ```

4. **Subsequent runs use fixed values:**
   ```bash
   uv run traenslenzor/doc_classifier/run.py \
       --config_path .configs/alexnet_scratch.toml
   # No tuning flags needed
   ```

### When to Re-Tune

**Learning Rate:**
- Switching models (AlexNet → ResNet → ViT)
- Changing optimizer (AdamW → SGD)
- Changing batch size significantly
- Fine-tuning vs training from scratch
- Different stage of training (e.g., after unfreezing backbone)

**Batch Size:**
- Switching GPUs (different memory)
- Changing model architecture
- Modifying image resolution
- Enabling/disabling mixed precision

## Advanced Usage

### Custom LR Finder Parameters

The `lr_find()` method accepts many parameters:

```python
lr_finder = tuner.lr_find(
    model,
    datamodule=datamodule,
    min_lr=1e-8,           # Minimum LR to test
    max_lr=1.0,            # Maximum LR to test
    num_training=100,      # Number of training iterations
    mode="exponential",    # or "linear"
    early_stop_threshold=4.0,  # Stop if loss > initial_loss * threshold
    update_attr=True,      # Automatically update model.hparams.lr
)
```

### LR Finder for Fine-Tuning

For fine-tuning, you might want lower LR range:

```python
# Fine-tuning typically needs lower LRs
lr_finder = tuner.lr_find(
    model,
    datamodule=datamodule,
    min_lr=1e-7,    # Lower minimum
    max_lr=1e-2,    # Lower maximum
    num_training=100,
)
```

### Using LR Finder Results

```python
# Get all results
results = lr_finder.results
# Contains: {'lr': [...], 'loss': [...]}

# Get suggestion
optimal_lr = lr_finder.suggestion()

# Manual analysis (if you disagree with suggestion)
import numpy as np
losses = np.array(results['loss'])
lrs = np.array(results['lr'])

# Find LR at minimum loss
min_loss_idx = np.argmin(losses)
lr_at_min = lrs[min_loss_idx]

# Or find LR at steepest gradient
gradients = np.gradient(losses)
steepest_idx = np.argmin(gradients)
lr_at_steepest = lrs[steepest_idx]
```

## Comparison: LR Finder vs Manual Tuning

| Aspect | Manual Tuning | LR Finder |
|--------|--------------|-----------|
| **Time** | Hours to days | 1-2 minutes |
| **Accuracy** | Hit or miss | Scientific (based on loss curve) |
| **Effort** | Many trial runs | Single automated run |
| **Cost** | Compute expensive | Minimal compute |
| **Reproducibility** | Hard to replicate | Fully automated |

## Best Practices

### 1. Order Matters

**Always run batch size tuning BEFORE learning rate finding:**

✅ **Correct:**
```bash
--tune-batch-size --tune-learning-rate  # BS first, then LR
```

❌ **Wrong:**
```bash
--tune-learning-rate --tune-batch-size  # LR first, then BS
```

**Why?** The optimal LR depends on the batch size. If you find LR with batch size 32, then increase to 64, the LR might not be optimal anymore.

### 2. Use OneCycleLR with Found LR

After finding the optimal LR, use it with OneCycleLR:

```toml
[module_config.optimizer]
learning_rate = 0.00316  # From lr_find

[module_config.scheduler]
max_lr = 0.01  # ~3x the base LR
div_factor = 10.0  # Start at max_lr/10 = 0.001
pct_start = 0.3
anneal_strategy = "cos"
```

### 3. Save the LR Plot

The LR finder plot is invaluable for understanding your model:

```bash
# After running with --tune-learning-rate
ls .logs/lr_finder_plot.png  # Check the plot
```

Open it to see:
- If the loss curve is smooth → good
- If it's noisy → might need more iterations
- If it never decreases → model/data issues
- If it plateaus early → might need higher max_lr

### 4. Don't Overtune

**For initial experiments:**
- Use tuner to find ballpark values
- Update configs
- Train without tuning

**For production:**
- Run tuning occasionally (e.g., when changing GPU)
- Not every single run

## Troubleshooting

### LR Finder Issues

**Problem:** "ValueError: Could not find lr or learning_rate in model"

**Solution:** Ensure your model has one of these:
```python
class MyModel(LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  # This creates self.hparams.learning_rate
        # or
        self.learning_rate = learning_rate
```

**Problem:** LR finder suggests very high LR (e.g., 1.0)

**Solution:**
- Model might be too simple for the data
- Check if loss curve actually decreases
- Try narrower LR range: `max_lr=1e-2`

**Problem:** LR finder suggests very low LR (e.g., 1e-7)

**Solution:**
- Model might be too complex or unstable
- Check for NaN/Inf in training
- Try: `min_lr=1e-6, max_lr=1e-3`

### Batch Size Finder Issues

**Problem:** "CUDA out of memory" even with small init_val

**Solution:**
- Reduce image resolution
- Enable mixed precision: `precision="16-mixed"`
- Use gradient accumulation instead

**Problem:** Found batch size is very small (e.g., 2 or 4)

**Solution:**
- Model is too large for GPU
- Use gradient accumulation:
  ```toml
  [datamodule_config]
  batch_size = 4

  [trainer_config]
  accumulate_grad_batches = 16  # Effective batch = 4*16 = 64
  ```

## References

- **LR Range Test Paper:** Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks"
- **OneCycleLR Paper:** Smith, L. N. (2018). "A disciplined approach to neural network hyper-parameters"
- **Lightning Docs:** https://lightning.ai/docs/pytorch/stable/tuning/profiler_expert.html

## Summary

| Feature | LR Finder | Batch Size Finder |
|---------|-----------|-------------------|
| **Method** | `tuner.lr_find()` | `tuner.scale_batch_size()` |
| **Purpose** | Find optimal learning rate | Find maximum batch size |
| **Duration** | ~1-2 minutes | ~1-2 minutes |
| **CLI Flag** | `--tune-learning-rate` | `--tune-batch-size` |
| **Output** | Suggested LR + plot | Optimal batch size |
| **When** | Every model/dataset change | Every GPU/resolution change |
| **Priority** | ⭐⭐⭐⭐⭐ (most important!) | ⭐⭐⭐ (nice to have) |

**Key Takeaway:** The `Tuner` is not just for batch size - its `lr_find()` method is arguably MORE important for achieving good training results!
