# Hyperparameter Tuning with TunerConfig

## Overview

The `TunerConfig` provides automated hyperparameter optimization using PyTorch Lightning's Tuner class. It supports two types of tuning:

1. **Batch Size Tuning**: Finds the optimal batch size that fits in GPU memory
2. **Learning Rate Tuning**: Runs LR Range Test to find optimal learning rate

Both methods run **BEFORE** training begins and should be executed in order: batch size first, then learning rate (since optimal LR depends on batch size).

## Architecture

### Key Components

- **`TunerConfig`** (`traenslenzor/doc_classifier/lightning/lit_tuning.py`):
  - Pydantic config class inheriting from `BaseConfig[Tuner]`
  - Configurable parameters for both batch size and LR tuning
  - `run_tuning()` method executes tuning logic

- **`ExperimentConfig`** (`traenslenzor/doc_classifier/configs/experiment_config.py`):
  - Contains `tuner_config: TunerConfig` field
  - Automatically calls `tuner_config.run_tuning()` before training

- **Removed Components**:
  - ❌ `BatchSizeFinder` callback (incorrect usage pattern)
  - ❌ `--tune-batch-size` and `--tune-learning-rate` CLI flags
  - ✅ Now controlled entirely via TOML config

## Configuration

### TOML Configuration

```toml
[tuner_config]
# Batch Size Tuning
use_batch_size_tuning = true
batch_size_mode = "power"           # 'power' or 'binsearch'
batch_size_init_val = 2             # Starting batch size
batch_size_max_trials = 25          # Max attempts

# Learning Rate Tuning
use_learning_rate_tuning = true
lr_min = 1.0e-8                     # Minimum LR
lr_max = 1.0                        # Maximum LR
lr_num_training = 100               # Iterations for LR Range Test
update_model_lr = true              # Auto-update model LR
verbose = true
is_debug = false
```

### TunerConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_batch_size_tuning` | bool | False | Enable batch size optimization |
| `batch_size_mode` | str | "power" | Search mode: 'power' (2^n) or 'binsearch' |
| `batch_size_init_val` | int | 2 | Initial batch size value |
| `batch_size_max_trials` | int | 25 | Maximum tuning iterations |
| `use_learning_rate_tuning` | bool | False | Enable LR Range Test |
| `lr_min` | float | 1e-8 | Minimum learning rate |
| `lr_max` | float | 1.0 | Maximum learning rate |
| `lr_num_training` | int | 100 | Training iterations for LR finder |
| `lr_plot_path` | Path \| None | None | Custom plot save path |
| `update_model_lr` | bool | True | Auto-update model with suggested LR |
| `verbose` | bool | True | Enable detailed logging |
| `is_debug` | bool | False | Enable debug output |

## Usage

### Basic Usage

1. **Enable tuning in your TOML config**:

```toml
[tuner_config]
use_batch_size_tuning = true
use_learning_rate_tuning = true
```

2. **Run training**:

```bash
uv run traenslenzor/doc_classifier/run.py --config_path .configs/your_config.toml
```

The tuner will automatically run before training starts.

### Example Workflow

```python
# Load config
config = ExperimentConfig.from_toml("my_config.toml")

# Tuning happens automatically in setup_target_and_run()
trainer = config.setup_target_and_run(stage="train")

# Optimal hyperparameters are now set:
# - datamodule.hparams.batch_size (if batch size tuning enabled)
# - module.params.optimizer.learning_rate (if LR tuning enabled)
```

### Tuning Order

**IMPORTANT**: Always run batch size tuning BEFORE learning rate tuning!

```toml
[tuner_config]
# ✅ CORRECT: Batch size first, then LR
use_batch_size_tuning = true
use_learning_rate_tuning = true
```

Why? The optimal learning rate depends on the batch size. If you find LR with batch_size=32, then change to batch_size=128, the LR will be suboptimal.

## How It Works

### Batch Size Tuning

1. **Binary/Power Search**: Tries progressively larger batch sizes
2. **Memory Monitoring**: Detects when GPU memory is exhausted
3. **Safety Margin**: Returns largest working batch size with headroom
4. **Auto-Update**: Updates `datamodule.hparams.batch_size`

**Example Output**:
```
[TunerConfig::run_tuning] ============================================================
[TunerConfig::run_tuning] BATCH SIZE TUNING
[TunerConfig::run_tuning] ============================================================
[TunerConfig::run_tuning] Mode: power
[TunerConfig::run_tuning] Initial value: 2
[TunerConfig::run_tuning] Max trials: 25
[TunerConfig::run_tuning] ✓ Optimal batch size found: 128
```

### Learning Rate Tuning (LR Range Test)

Based on Leslie Smith's LR Range Test paper:

1. **Exponential Increase**: Trains for ~100 iterations while exponentially increasing LR from `lr_min` to `lr_max`
2. **Loss Tracking**: Records loss at each LR value
3. **Steepest Descent**: Finds LR with steepest negative gradient in loss curve
4. **Auto-Update**: Updates `module.params.optimizer.learning_rate`
5. **Plot Generation**: Saves loss vs LR plot to `.logs/lr_finder_plot.png`

**Example Output**:
```
[TunerConfig::run_tuning] ============================================================
[TunerConfig::run_tuning] LEARNING RATE FINDER (LR Range Test)
[TunerConfig::run_tuning] ============================================================
[TunerConfig::run_tuning] LR range: [1.00e-08, 1.00e+00]
[TunerConfig::run_tuning] Training iterations: 100
[TunerConfig::run_tuning] Training while exponentially increasing LR...
[TunerConfig::run_tuning] ✓ Suggested learning rate: 3.16e-03
[TunerConfig::run_tuning]   Updated optimizer LR: 1.00e-03 → 3.16e-03
[TunerConfig::run_tuning]   LR finder plot saved to: /path/to/.logs/lr_finder_plot.png
```

### LR Finder Plot

The plot shows:
- **X-axis**: Learning rate (log scale)
- **Y-axis**: Training loss
- **Suggested LR**: Marked with vertical line
- **Interpretation**: Choose LR at steepest descent (fastest loss decrease)

## Best Practices

### When to Use Tuning

✅ **Use tuning when**:
- Starting a new project
- Training on new hardware (different GPU)
- Significantly changing model architecture
- Changing dataset size or characteristics

❌ **Don't use tuning when**:
- Fine-tuning from checkpoint (hyperparameters already optimized)
- Running quick experiments (tuning adds 2-5 minutes)
- Resuming interrupted training

### Recommended Settings

**Quick tuning** (1-2 minutes):
```toml
[tuner_config]
use_batch_size_tuning = true
batch_size_max_trials = 10
use_learning_rate_tuning = true
lr_num_training = 50
```

**Thorough tuning** (3-5 minutes):
```toml
[tuner_config]
use_batch_size_tuning = true
batch_size_max_trials = 25
use_learning_rate_tuning = true
lr_num_training = 100
```

**Production settings** (after tuning once):
```toml
[tuner_config]
# Disable tuning, use fixed values
use_batch_size_tuning = false
use_learning_rate_tuning = false

[datamodule_config]
batch_size = 128  # From previous tuning

[module_config.optimizer]
learning_rate = 3.16e-3  # From previous tuning
```

### Integration with Schedulers

If using OneCycleLR scheduler:

1. **Run LR tuning first** to get base LR
2. **Set `max_lr`** manually in config (typically 3-10x base LR)
3. **Disable `update_model_lr`** if you want manual control:

```toml
[tuner_config]
use_learning_rate_tuning = true
update_model_lr = false  # Don't auto-update, just log suggestion

[module_config.optimizer]
learning_rate = 1.0e-3  # Set manually based on tuner suggestion

[module_config.scheduler]
max_lr = 3.0e-3  # 3x base_lr
```

## Troubleshooting

### Batch Size Tuning Fails

**Problem**: "CUDA out of memory" during tuning

**Solutions**:
- Reduce `batch_size_init_val` to 1 or 2
- Set `batch_size_mode = "binsearch"` for more conservative search
- Reduce `batch_size_max_trials` to avoid edge cases

### LR Tuning Gives Unstable Results

**Problem**: Suggested LR seems too high/low

**Solutions**:
- Increase `lr_num_training` for more stable suggestions (try 200-300)
- Narrow the search range: `lr_min = 1e-6`, `lr_max = 0.1`
- Manually inspect the plot at `.logs/lr_finder_plot.png`
- Look for the steepest descent point, not the minimum loss

### Tuning Takes Too Long

**Problem**: Tuning adds 10+ minutes before training

**Solutions**:
- Reduce `lr_num_training` to 50
- Set `batch_size_max_trials = 10`
- Run tuning once, then disable it and use fixed values

## Migration Guide

### From Old CLI Flags to TunerConfig

**OLD** (deprecated):
```bash
uv run run.py --config_path my.toml --tune-batch-size --tune-learning-rate
```

**NEW**:
```toml
# In my.toml
[tuner_config]
use_batch_size_tuning = true
use_learning_rate_tuning = true
```

```bash
uv run run.py --config_path my.toml
```

### From BatchSizeFinder Callback

**OLD** (incorrect):
```toml
[trainer_config.callbacks]
use_batch_size_finder = true
batch_size_mode = "power"
batch_size_init_val = 32
batch_size_max_trials = 25
```

**NEW** (correct):
```toml
[tuner_config]
use_batch_size_tuning = true
batch_size_mode = "power"
batch_size_init_val = 2
batch_size_max_trials = 25
```

### Key Differences

| Aspect | Old (BatchSizeFinder) | New (TunerConfig) |
|--------|----------------------|-------------------|
| **When runs** | During training loop | Before training starts |
| **Location** | Callback in trainer | Standalone config |
| **Time cost** | Full epoch (~10 min) | Quick test (~1 min) |
| **LR tuning** | Not supported | Supported via `lr_find()` |
| **Config location** | `trainer_config.callbacks` | `tuner_config` |

## Advanced Usage

### Custom LR Plot Path

```toml
[tuner_config]
use_learning_rate_tuning = true
lr_plot_path = "custom_plots/my_lr_finder.png"
```

### Programmatic Access

```python
from traenslenzor.doc_classifier import ExperimentConfig

config = ExperimentConfig.from_toml("my_config.toml")

# Access tuner config
tuner_cfg = config.tuner_config

# Modify at runtime
tuner_cfg.lr_min = 1e-6
tuner_cfg.lr_max = 0.1

# Run manually
trainer, module, datamodule = config.setup_target()
optimal_bs, suggested_lr = tuner_cfg.run_tuning(trainer, module, datamodule)

print(f"Optimal batch size: {optimal_bs}")
print(f"Suggested learning rate: {suggested_lr}")
```

## See Also

- **LR Range Test Paper**: [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) (Leslie Smith, 2015)
- **PyTorch Lightning Tuner**: [Official Docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html)
- **BATCH_SIZE_TUNING.md**: Why BatchSizeFinder callback was incorrect
- **TUNER_GUIDE.md**: Previous CLI-based tuning guide (deprecated)
