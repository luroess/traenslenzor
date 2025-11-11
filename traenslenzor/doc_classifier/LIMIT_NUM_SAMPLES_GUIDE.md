# Dataset Size Limiting Guide

## Overview

The `DocDataModule` supports the `limit_num_samples` parameter to reduce dataset sizes for **debugging, development, and quick experimentation**. This feature allows you to work with a subset of the full RVL-CDIP dataset without downloading or processing the entire dataset.

## Features

✅ **Absolute Limiting** - Specify exact number of samples (e.g., `limit_num_samples=1000`)
✅ **Fractional Limiting** - Specify percentage of dataset (e.g., `limit_num_samples=0.1` for 10%)
✅ **Automatic Application** - Applied uniformly across train, validation, and test splits
✅ **Cached After Loading** - Dataset loaded once, then cached for subsequent access
✅ **Memory Efficient** - Uses HuggingFace's `Dataset.select()` for efficient subsetting

## Configuration

### Absolute Number of Samples

Specify an exact number of samples to use from each split:

```python
from traenslenzor.doc_classifier.lightning.lit_datamodule import DocDataModuleConfig

config = DocDataModuleConfig(
    limit_num_samples=1000,  # Use exactly 1000 samples from each split
    batch_size=32,
)
```

**Use cases:**
- Quick smoke tests: `limit_num_samples=100`
- Development iteration: `limit_num_samples=1000`
- Small-scale experiments: `limit_num_samples=5000`

### Fractional Percentage

Specify a fraction of the dataset to use (between 0 and 1):

```python
config = DocDataModuleConfig(
    limit_num_samples=0.1,  # Use 10% of each split
    batch_size=32,
)
```

**Use cases:**
- Quick prototyping: `limit_num_samples=0.01` (1%)
- Medium experiments: `limit_num_samples=0.1` (10%)
- Large subset: `limit_num_samples=0.5` (50%)

### No Limit (Production)

For full dataset training, set to `None` (default):

```python
config = DocDataModuleConfig(
    limit_num_samples=None,  # Use full dataset (default)
    batch_size=32,
)
```

## How It Works

### Implementation

When `limit_num_samples` is set, the `DocDataModule` applies the limit **after loading** each dataset:

```python
def _ensure_dataset(self, stage: Stage) -> None:
    attr_name, cfg = self._stage_attr_map[stage]
    dataset = getattr(self, attr_name)
    if dataset is None:
        dataset = cfg.setup_target()

        # Apply limit_num_samples if configured
        if self.config.limit_num_samples is not None:
            dataset = self._apply_sample_limit(dataset, stage)

        setattr(self, attr_name, dataset)
```

### Calculation Logic

```python
def _apply_sample_limit(self, dataset: HFDataset, stage: Stage) -> HFDataset:
    limit = self.config.limit_num_samples
    original_size = len(dataset)

    # Calculate target size
    if isinstance(limit, float):
        # Fraction of dataset (e.g., 0.1 = 10%)
        target_size = int(original_size * limit)
    else:
        # Absolute number of samples
        target_size = min(limit, original_size)

    # Select subset
    limited_dataset = dataset.select(range(target_size))

    # Log limitation
    console.log(
        f"Limited {stage} dataset: {original_size} → {target_size} samples "
        f"({target_size/original_size*100:.1f}%)"
    )

    return limited_dataset
```

## Examples

### Example 1: Quick Smoke Test

Test training loop with minimal data:

```python
from traenslenzor.doc_classifier.configs.experiment_config import ExperimentConfig

config = ExperimentConfig(
    datamodule_config={
        "limit_num_samples": 100,  # Only 100 samples per split
        "batch_size": 16,
    },
    trainer_config={
        "max_epochs": 1,
        "fast_dev_run": False,
    },
)

trainer, module, datamodule = config.setup_target()
trainer.fit(module, datamodule)
```

**Expected behavior:**
- Train dataset: 100 samples
- Val dataset: 100 samples
- Test dataset: 100 samples
- Fast iteration for debugging

### Example 2: Development with 10% Dataset

Experiment with realistic subset:

```python
config = ExperimentConfig(
    datamodule_config={
        "limit_num_samples": 0.1,  # 10% of each split
        "batch_size": 32,
    },
    trainer_config={
        "max_epochs": 10,
    },
)

trainer, module, datamodule = config.setup_target()
trainer.fit(module, datamodule)
```

**Expected behavior:**
- Train dataset: ~32,000 samples (10% of 320,000)
- Val dataset: ~4,000 samples (10% of 40,000)
- Test dataset: ~4,000 samples (10% of 40,000)
- Reasonable training time for experimentation

### Example 3: Full Dataset Training

Production training without limits:

```python
config = ExperimentConfig(
    datamodule_config={
        "limit_num_samples": None,  # Full dataset
        "batch_size": 64,
    },
    trainer_config={
        "max_epochs": 100,
    },
)

trainer, module, datamodule = config.setup_target()
trainer.fit(module, datamodule)
```

**Expected behavior:**
- Train dataset: 320,000 samples (full)
- Val dataset: 40,000 samples (full)
- Test dataset: 40,000 samples (full)

## RVL-CDIP Dataset Sizes

For reference, the full RVL-CDIP dataset contains:

| Split      | Full Size | 10% (0.1) | 5% (0.05) | 1% (0.01) | 100 samples |
|------------|-----------|-----------|-----------|-----------|-------------|
| **Train**  | 320,000   | 32,000    | 16,000    | 3,200     | 100         |
| **Val**    | 40,000    | 4,000     | 2,000     | 400       | 100         |
| **Test**   | 40,000    | 4,000     | 2,000     | 400       | 100         |

## Validation and Error Handling

### Valid Values

```python
# ✅ Valid absolute limits
limit_num_samples=100        # Use 100 samples
limit_num_samples=1000       # Use 1000 samples
limit_num_samples=1_000_000  # Use 1M samples (capped to dataset size)

# ✅ Valid fractional limits
limit_num_samples=0.01   # 1%
limit_num_samples=0.1    # 10%
limit_num_samples=0.5    # 50%
limit_num_samples=1.0    # 100% (full dataset)

# ✅ No limit (default)
limit_num_samples=None
```

### Invalid Values

```python
# ❌ Invalid: float > 1.0
limit_num_samples=1.5
# ValueError: limit_num_samples as float must be in (0, 1], got 1.5

# ❌ Invalid: float <= 0
limit_num_samples=0.0
# ValueError: limit_num_samples as float must be in (0, 1], got 0.0

# ❌ Invalid: negative int
limit_num_samples=-100
# ValueError: Invalid target_size=-100 from limit_num_samples=-100

# ❌ Invalid: zero
limit_num_samples=0
# ValueError: Invalid target_size=0 from limit_num_samples=0
```

## Logging Output

When `verbose=True`, the datamodule logs the limitation:

```
[DocDataModule::_apply_sample_limit]: Limited train dataset: 320000 → 32000 samples (10.0%)
[DocDataModule::_apply_sample_limit]: Limited val dataset: 40000 → 4000 samples (10.0%)
[DocDataModule::_apply_sample_limit]: Limited test dataset: 40000 → 4000 samples (10.0%)
```

## Integration with Debug Mode

The `limit_num_samples` parameter works seamlessly with `is_debug` mode:

```python
config = ExperimentConfig(
    is_debug=True,  # Enables debug-specific settings
    datamodule_config={
        "limit_num_samples": 100,  # Small dataset for debugging
    },
)
```

Debug mode automatically sets:
- `num_workers=0` (single-threaded loading)
- Verbose logging enabled
- Combined with `limit_num_samples` for fast debugging iterations

## Best Practices

### 1. Development Workflow

Start small and scale up:

```python
# Phase 1: Smoke test (seconds)
config.datamodule_config.limit_num_samples = 100

# Phase 2: Quick iteration (minutes)
config.datamodule_config.limit_num_samples = 1000

# Phase 3: Realistic testing (hours)
config.datamodule_config.limit_num_samples = 0.1  # 10%

# Phase 4: Final training (overnight)
config.datamodule_config.limit_num_samples = None  # Full dataset
```

### 2. Hyperparameter Tuning

Use moderate limits for Optuna sweeps:

```python
config = ExperimentConfig(
    datamodule_config={
        "limit_num_samples": 0.05,  # 5% for fast trials
    },
    optuna_config={
        "n_trials": 50,
        "timeout": 3600,  # 1 hour
    },
)

config.run_optuna_study()
```

### 3. CI/CD Testing

Use minimal datasets for automated tests:

```python
# In CI pipeline
config = ExperimentConfig(
    datamodule_config={
        "limit_num_samples": 50,  # Tiny dataset
        "batch_size": 16,
    },
    trainer_config={
        "max_epochs": 1,
        "fast_dev_run": True,
    },
)
```

## Performance Considerations

### Memory Usage

- `limit_num_samples` uses `Dataset.select()` which creates a **view**, not a copy
- Memory footprint is proportional to the **limited size**, not the full dataset
- Safe to use with large limits without memory concerns

### Loading Time

- Initial dataset loading time is **not affected** by `limit_num_samples`
- HuggingFace loads dataset metadata first, then applies selection
- For true streaming/lazy loading, consider `streaming=True` in `RVLCDIPConfig`

### Caching

- Limited datasets are **cached** after first access
- Subsequent calls to `train_dataloader()`, `val_dataloader()`, etc. use cached data
- Changing `limit_num_samples` requires creating a new `DocDataModule` instance

## Troubleshooting

### Issue: "ValueError: limit_num_samples must be in (0, 1]"

**Cause:** Float value outside valid range

**Solution:**
```python
# ❌ Wrong
limit_num_samples=1.5

# ✅ Correct
limit_num_samples=0.5  # 50%
```

### Issue: Dataset still takes long to load

**Cause:** HuggingFace downloads full dataset first

**Solution:** Consider using `streaming=True` for true lazy loading:
```python
config.datamodule_config.train_ds.streaming = True
```

### Issue: Not enough samples for batch size

**Cause:** `limit_num_samples` too small for configured batch size

**Solution:** Adjust batch size or increase limit:
```python
# If limit_num_samples=50 and batch_size=64
# ❌ Only 0 full batches possible

# ✅ Solution 1: Reduce batch size
batch_size=16

# ✅ Solution 2: Increase limit
limit_num_samples=128
```

## Related Configuration

- **`batch_size`**: Number of samples per batch
- **`num_workers`**: Worker processes for data loading
- **`is_debug`**: Debug mode (auto-sets `num_workers=0`)
- **`verbose`**: Enable logging output
- **`streaming`**: Lazy dataset loading (in `RVLCDIPConfig`)

## References

- **HuggingFace `Dataset.select()`**: https://huggingface.co/docs/datasets/process#select-and-filter
- **RVL-CDIP Dataset**: https://huggingface.co/datasets/chainyo/rvl-cdip
- **PyTorch Lightning DataModule**: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
