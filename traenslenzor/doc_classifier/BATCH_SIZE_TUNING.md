# Batch Size Tuning - Correct Usage

## The Problem

When `use_batch_size_finder = true` was set in the config, the training ran for a full epoch (10+ minutes) before crashing, instead of quickly finding the optimal batch size.

## Root Cause

The **`BatchSizeFinder` callback** in PyTorch Lightning is NOT meant to be used as a regular callback during training. It runs during the training loop itself, which is why it trained for a full epoch.

According to Lightning documentation, there are two correct ways to find optimal batch size:

### ❌ WRONG Way (What We Were Doing)

```python
# Adding BatchSizeFinder as a callback
trainer = Trainer(callbacks=[BatchSizeFinder(...)])
trainer.fit(model)  # This runs a full training loop!
```

### ✅ CORRECT Way

**Option 1: Use the `Tuner` class BEFORE training**

```python
from lightning.pytorch.tuner import Tuner

trainer = Trainer(...)
tuner = Tuner(trainer)

# This runs BEFORE trainer.fit() and quickly finds optimal batch size
tuner.scale_batch_size(model, datamodule=datamodule, mode="power")

# Now train with the optimized batch size
trainer.fit(model, datamodule=datamodule)
```

**Option 2: Custom callback for specific epochs (advanced)**

```python
class FineTuneBatchSizeFinder(BatchSizeFinder):
    def on_fit_start(self, *args, **kwargs):
        return  # Disable automatic triggering

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in [0, 5, 10]:
            self.scale_batch_size(trainer, pl_module)
```

## The Fix

### 1. Updated `ExperimentConfig.setup_target_and_run()`

Added `tune_batch_size` parameter that uses the `Tuner` class:

```python
def setup_target_and_run(
    self,
    stage: Stage | str | None = None,
    tune_batch_size: bool = False,  # NEW parameter
) -> Trainer:
    """Run experiment with optional batch size tuning BEFORE training."""

    # ... setup code ...

    # Run batch size tuning BEFORE training if requested
    if tune_batch_size and resolved_stage is Stage.TRAIN:
        from pytorch_lightning.tuner import Tuner

        tuner = Tuner(trainer)
        tuner.scale_batch_size(
            lit_module,
            datamodule=lit_datamodule,
            mode=mode,
            init_val=init_val,
            max_trials=max_trials,
        )

    # Now train with optimized batch size
    trainer.fit(lit_module, datamodule=lit_datamodule)
```

### 2. Updated CLI (`run.py`)

Added `--tune-batch-size` flag:

```python
# Extract from CLI
tune_batch_size = False
for arg in sys.argv[1:]:
    if arg in ("--tune-batch-size", "--tune_batch_size"):
        tune_batch_size = True

# Run with tuning
config.setup_target_and_run(tune_batch_size=tune_batch_size)
```

### 3. Disabled `BatchSizeFinder` Callback

Updated all config files to set `use_batch_size_finder = false` by default.

## New Usage

### Find Optimal Batch Size (One-Time)

```bash
# Run batch size tuning BEFORE training
uv run traenslenzor/doc_classifier/run.py \
    --config_path .configs/alexnet_scratch.toml \
    --tune-batch-size
```

**What happens:**
1. `Tuner.scale_batch_size()` runs quick tests with increasing batch sizes
2. Finds the largest batch size that fits in GPU memory
3. Updates `datamodule.hparams.batch_size` automatically
4. Prints: `"Optimal batch size found: 128"` (example)
5. Starts normal training with the optimized batch size

**Duration:** ~1-2 minutes (much faster than full epoch!)

### Normal Training (With Known Batch Size)

```bash
# Just train with the batch size specified in config
uv run traenslenzor/doc_classifier/run.py \
    --config_path .configs/alexnet_scratch.toml
```

## Recommended Workflow

### Initial Setup (Per Model/GPU)

1. **Enable batch size tuning for first run:**
   ```bash
   uv run traenslenzor/doc_classifier/run.py \
       --config_path .configs/alexnet_scratch.toml \
       --tune-batch-size
   ```

2. **Note the optimal batch size from logs:**
   ```
   [ExperimentConfig::tuner]: Optimal batch size found: 128
   ```

3. **Update your config file:**
   ```toml
   [datamodule_config]
   batch_size = 128  # Use the found optimal value
   ```

4. **Subsequent runs use the fixed batch size:**
   ```bash
   uv run traenslenzor/doc_classifier/run.py \
       --config_path .configs/alexnet_scratch.toml
   # No --tune-batch-size flag needed
   ```

### When to Re-Tune

- Switching to different GPU
- Changing model architecture (AlexNet → ResNet → ViT)
- Modifying image resolution in transforms
- Enabling gradient accumulation or mixed precision

## Technical Details

### How `Tuner.scale_batch_size()` Works

1. **Power Mode (default):**
   - Starts at `init_val` (e.g., 2)
   - Doubles batch size: 2 → 4 → 8 → 16 → 32 → 64 → 128 → ...
   - Runs a few batches at each size
   - Stops when OOM error occurs
   - Returns the last successful batch size

2. **Binary Search Mode:**
   - More precise but slower
   - Finds exact optimal batch size via binary search
   - Use when GPU memory is very limited

### What Gets Updated

The tuner automatically updates:
```python
datamodule.hparams.batch_size = optimal_value
datamodule.config.batch_size = optimal_value
```

This is why we needed `save_hyperparameters()` in `DocDataModule.__init__()`.

## Config Settings

The batch size tuning parameters are still in the config, but used differently:

```toml
[trainer_config.callbacks]
# Don't use as callback - used by Tuner instead
use_batch_size_finder = false  # MUST be false

# These settings are read by setup_target_and_run() when tune_batch_size=True
batch_size_mode = "power"        # "power" or "binsearch"
batch_size_init_val = 2          # Starting batch size for search
batch_size_max_trials = 25       # Maximum iterations
```

## Summary

| Aspect | Old (Broken) | New (Correct) |
|--------|-------------|---------------|
| **Method** | `BatchSizeFinder` callback | `Tuner.scale_batch_size()` |
| **When** | During training loop | Before `trainer.fit()` |
| **Duration** | Full epoch (~10+ min) | Quick test (~1-2 min) |
| **Usage** | `use_batch_size_finder=true` | `--tune-batch-size` flag |
| **Config** | Set in callbacks | Disabled in callbacks |

The key insight: **Batch size tuning is a PRE-PROCESSING step, not a training callback!**
