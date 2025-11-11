# Tuner Refactoring Summary

## Overview

Refactored PyTorch Lightning Tuner functionality from inline code in `ExperimentConfig` into a dedicated, configurable `TunerConfig` module. This improves modularity, maintainability, and configuration flexibility.

## Changes Made

### 1. Created `TunerConfig` (`lit_tuning.py`)

**Location**: `traenslenzor/doc_classifier/lightning/lit_tuning.py`

**Key Features**:
- Pydantic config class inheriting from `BaseConfig[Tuner]`
- Configurable parameters for both batch size and learning rate tuning
- `run_tuning()` method encapsulates all tuning logic
- Automatic LR plot generation with configurable save path
- Clean separation of concerns

**Configuration Fields**:
```python
class TunerConfig(BaseConfig[Tuner]):
    # Batch size tuning
    use_batch_size_tuning: bool = False
    batch_size_mode: str = "power"
    batch_size_init_val: int = 2
    batch_size_max_trials: int = 25

    # Learning rate tuning
    use_learning_rate_tuning: bool = False
    lr_min: float = 1e-8
    lr_max: float = 1.0
    lr_num_training: int = 100
    lr_plot_path: Path | None = None
    update_model_lr: bool = True

    # Logging
    verbose: bool = True
    is_debug: bool = False
```

### 2. Updated `ExperimentConfig`

**Changes**:
- Added `tuner_config: TunerConfig` field
- Removed `tune_batch_size` and `tune_learning_rate` parameters from `setup_target_and_run()`
- Simplified method to call `self.tuner_config.run_tuning()` when tuning is enabled
- All tuning configuration now in TOML file, not CLI flags

**Before**:
```python
def setup_target_and_run(
    self,
    stage: Stage | str | None = None,
    tune_batch_size: bool = False,
    tune_learning_rate: bool = False,
) -> Trainer:
    # ... 80+ lines of tuning code inline ...
```

**After**:
```python
def setup_target_and_run(
    self,
    stage: Stage | str | None = None,
) -> Trainer:
    # ... setup ...

    if self.tuner_config.use_batch_size_tuning or self.tuner_config.use_learning_rate_tuning:
        self.tuner_config.run_tuning(trainer, lit_module, lit_datamodule)

    # ... training ...
```

### 3. Removed BatchSizeFinder Callback

**File**: `lit_trainer_callbacks.py`

**Removed**:
- Import of `BatchSizeFinder` from `pytorch_lightning.callbacks`
- Configuration fields: `use_batch_size_finder`, `batch_size_mode`, `batch_size_init_val`, `batch_size_max_trials`
- Callback instantiation in `setup_target()`

**Reason**: BatchSizeFinder callback runs **during** training (incorrect), while Tuner runs **before** training (correct).

### 4. Updated `run.py` CLI

**Removed**:
- `--tune-batch-size` / `--tune_batch_size` CLI flag parsing
- `--tune-learning-rate` / `--tune_learning_rate` / `--tune-lr` CLI flag parsing
- Passing tuning flags to `setup_target_and_run()`

**New Approach**: All tuning configuration is in TOML config under `[tuner_config]` section.

**Before**:
```bash
uv run run.py --config_path my.toml --tune-batch-size --tune-lr
```

**After**:
```toml
# In my.toml
[tuner_config]
use_batch_size_tuning = true
use_learning_rate_tuning = true
```
```bash
uv run run.py --config_path my.toml
```

### 5. Updated Lightning Module Exports

**File**: `lightning/__init__.py`

**Added**: `TunerConfig` to exports

```python
from .lit_tuning import TunerConfig

__all__ = [
    # ... existing exports ...
    "TunerConfig",
]
```

### 6. Updated Example Configs

**Files Updated**:
- `.configs/cli_test.toml` - Removed deprecated `use_batch_size_finder` fields, added `[tuner_config]` section
- `.configs/tuning_example.toml` - **NEW** comprehensive example showing tuning usage

### 7. Documentation

**Created**:
- `TUNER_CONFIG_GUIDE.md` - Comprehensive 300+ line guide covering:
  - Architecture overview
  - Configuration reference
  - Usage examples
  - Best practices
  - Migration guide from old approach
  - Troubleshooting

## Benefits

### 1. **Modularity**
- Tuning logic isolated in dedicated module
- Can be tested independently
- Easier to extend with new tuning methods

### 2. **Configuration-Driven**
- All tuning parameters in TOML config (declarative)
- No CLI flag parsing required
- Version-controlled tuning strategies
- Easy to share configs between experiments

### 3. **Maintainability**
- Removed 80+ lines of inline code from `ExperimentConfig`
- Clear separation of concerns
- Single source of truth for tuning parameters

### 4. **Flexibility**
- Easy to customize LR plot save path
- Can enable/disable tuning per experiment
- Fine-grained control over tuning parameters
- Programmatic access via `tuner_config` field

### 5. **Type Safety**
- All tuning parameters are typed Pydantic fields
- Validation happens at config load time
- IDE autocomplete for all fields

## Migration Path

### For Users

**Minimal Changes Required**:

1. **Update TOML configs**: Add `[tuner_config]` section instead of using CLI flags
2. **Remove deprecated fields**: Delete `use_batch_size_finder` from `[trainer_config.callbacks]`
3. **Update CLI commands**: Remove `--tune-batch-size` and `--tune-lr` flags

### For Developers

**Code Changes**:

1. **Import TunerConfig**: Available from `traenslenzor.doc_classifier.lightning`
2. **Access via config**: `experiment_config.tuner_config.run_tuning(...)`
3. **Remove BatchSizeFinder**: No longer in callbacks

## Testing

**Verification**:
```bash
# Import test
uv run python -c "from traenslenzor.doc_classifier.lightning import TunerConfig; print('✓ Success')"

# Config test
uv run python -c "from traenslenzor.doc_classifier import ExperimentConfig; \
    config = ExperimentConfig(); print('✓ Has tuner_config:', hasattr(config, 'tuner_config'))"

# Full integration test
uv run traenslenzor/doc_classifier/run.py --config_path .configs/tuning_example.toml
```

## File Summary

### Modified Files
- `traenslenzor/doc_classifier/lightning/lit_tuning.py` - **NEW** (180 lines)
- `traenslenzor/doc_classifier/lightning/__init__.py` - Added TunerConfig export
- `traenslenzor/doc_classifier/lightning/lit_trainer_callbacks.py` - Removed BatchSizeFinder
- `traenslenzor/doc_classifier/configs/experiment_config.py` - Added tuner_config field, simplified setup_target_and_run
- `traenslenzor/doc_classifier/run.py` - Removed CLI flag parsing
- `.configs/cli_test.toml` - Removed deprecated fields, added [tuner_config]
- `.configs/tuning_example.toml` - **NEW** example config
- `traenslenzor/doc_classifier/TUNER_CONFIG_GUIDE.md` - **NEW** comprehensive documentation

### Lines of Code
- **Removed**: ~130 lines (inline tuning code + BatchSizeFinder + CLI parsing)
- **Added**: ~180 lines (TunerConfig class + documentation comments)
- **Net Change**: +50 lines (better organized, more maintainable)

## Backward Compatibility

### Breaking Changes
⚠️ **CLI flags removed**: `--tune-batch-size` and `--tune-learning-rate`

**Migration**: Use `[tuner_config]` section in TOML instead

### Non-Breaking Changes
✅ **Existing configs**: Will work with `use_batch_size_tuning = false` (default)
✅ **API**: `ExperimentConfig.setup_target_and_run()` still works (just no tuning parameters)

## Future Enhancements

Potential improvements now that tuning is modular:

1. **Additional Tuning Methods**:
   - Gradient accumulation tuning
   - Mixed precision tuning
   - Number of workers optimization

2. **Advanced LR Finding**:
   - Multiple LR finder strategies (Smith, Fast.ai, etc.)
   - Custom suggestion algorithms
   - Multi-stage LR finding

3. **Tuning Profiles**:
   - Predefined tuning profiles (quick/thorough/production)
   - Model-specific tuning strategies
   - Auto-tuning based on hardware detection

4. **Integration**:
   - W&B logging of tuning results
   - Optuna integration for tuning hyperparameters
   - Checkpoint-based tuning resume

## References

- **PyTorch Lightning Tuner**: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html
- **LR Range Test**: https://arxiv.org/abs/1506.01186 (Leslie Smith, 2015)
- **Config-as-Factory Pattern**: `traenslenzor/doc_classifier/utils/base_config.py`
