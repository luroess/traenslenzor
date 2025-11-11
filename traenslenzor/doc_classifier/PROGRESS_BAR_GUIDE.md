# Progress Bar Customization Guide

## Quick Start

Our custom progress bars (`CustomTQDMProgressBar` and `CustomRichProgressBar`) automatically hide the version number (`v_num`) that PyTorch Lightning adds by default.

### Before
```
Epoch 0: 57%|███| 2875/5000 [03:52<02:51, 12.38it/s, v_num=2hcj, train/loss=1.010]
```

### After
```
Epoch 0: 57%|███| 2875/5000 [03:52<02:51, 12.38it/s, train/loss=1.010]
```

## Adding Metrics to Progress Bar

Use `self.log(..., prog_bar=True)` in your `LightningModule` with our standardized `Metric` enum:

```python
from traenslenzor.doc_classifier.utils.schemas import Metric

class MyLightningModule(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        acc = self.compute_accuracy(batch)

        # Log to progress bar using Metric enum
        self.log(Metric.TRAIN_LOSS, loss, prog_bar=True)
        self.log(Metric.TRAIN_ACCURACY, acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        acc = self.compute_accuracy(batch)

        # Validation metrics
        self.log(Metric.VAL_LOSS, loss, prog_bar=True)
        self.log(Metric.VAL_ACCURACY, acc, prog_bar=True)
```

## Available Metrics

From `traenslenzor.doc_classifier.utils.schemas.Metric`:

### Training Metrics
- `Metric.TRAIN_LOSS` → `"train/loss"`
- `Metric.TRAIN_ACCURACY` → `"train/accuracy"`

### Validation Metrics
- `Metric.VAL_LOSS` → `"val/loss"`
- `Metric.VAL_ACCURACY` → `"val/accuracy"`
- `Metric.VAL_CONFUSION_MATRIX` → `"val/confusion_matrix"`

### Test Metrics
- `Metric.TEST_LOSS` → `"test/loss"`
- `Metric.TEST_ACCURACY` → `"test/accuracy"`

## Progress Bar Types

### TQDM Progress Bar (Default)

```toml
[callbacks]
use_tqdm_progress_bar = true
use_rich_progress_bar = false
tqdm_refresh_rate = 1  # Update every batch
```

**Output:**
```
Epoch 0: 57%|███████████████████████▌                    | 2875/5000 [03:52<02:51, 12.38it/s, train/loss=1.010, train/accuracy=0.65]
```

### Rich Progress Bar (Enhanced)

```toml
[callbacks]
use_tqdm_progress_bar = false
use_rich_progress_bar = true
```

**Output:**
```
Epoch 0 ━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57% 0:03:52 < 0:02:51
train/loss: 1.010 • train/accuracy: 0.65
```

## Advanced: Custom Metric Filtering

If you want to customize which metrics appear in the progress bar, create a custom subclass:

### Example: Show Only Loss Metrics

```python
from traenslenzor.doc_classifier.lightning.lit_trainer_callbacks import CustomTQDMProgressBar
from traenslenzor.doc_classifier.utils.schemas import Metric

class LossOnlyProgressBar(CustomTQDMProgressBar):
    """Progress bar that only shows loss metrics."""

    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)

        # Remove accuracy metrics
        items.pop(Metric.TRAIN_ACCURACY, None)
        items.pop(Metric.VAL_ACCURACY, None)
        items.pop(Metric.TEST_ACCURACY, None)

        return items
```

### Example: Whitelist Specific Metrics

```python
class MinimalProgressBar(CustomTQDMProgressBar):
    """Progress bar showing only essential metrics."""

    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)

        # Only show these metrics
        allowed = {
            Metric.TRAIN_LOSS,
            Metric.VAL_LOSS,
        }

        return {k: v for k, v in items.items() if k in allowed}
```

### Example: Format Metric Values

```python
class FormattedProgressBar(CustomTQDMProgressBar):
    """Progress bar with custom metric formatting."""

    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)

        # Round all metrics to 3 decimal places
        return {k: round(v, 3) for k, v in items.items()}
```

## Using Custom Progress Bars

To use your custom progress bar, modify the `TrainerCallbacksConfig.setup_target()` method:

```python
# In lit_trainer_callbacks.py
if self.use_tqdm_progress_bar:
    callbacks.append(
        LossOnlyProgressBar(refresh_rate=self.tqdm_refresh_rate),  # Your custom class
    )
```

Or create a new config field for custom progress bar classes and use dependency injection.

## Logging Options

The `self.log()` method has several useful parameters:

```python
self.log(
    Metric.TRAIN_LOSS,
    loss,
    prog_bar=True,      # Show in progress bar
    logger=True,        # Log to W&B/TensorBoard
    on_step=True,       # Log at each step
    on_epoch=True,      # Log epoch average
    sync_dist=True,     # Sync across distributed processes
)
```

### Common Patterns

**Step-level metrics (training):**
```python
self.log(Metric.TRAIN_LOSS, loss, prog_bar=True, on_step=True, on_epoch=False)
```

**Epoch-level metrics (validation):**
```python
self.log(Metric.VAL_LOSS, loss, prog_bar=True, on_step=False, on_epoch=True)
```

**Both step and epoch:**
```python
self.log(Metric.TRAIN_LOSS, loss, prog_bar=True, on_step=True, on_epoch=True)
# Creates two metrics: "train/loss_step" and "train/loss_epoch"
```

## References

- **Custom Progress Bars:** `traenslenzor/doc_classifier/lightning/lit_trainer_callbacks.py`
- **Metric Enum:** `traenslenzor/doc_classifier/utils/schemas.py`
- **Config:** `TrainerCallbacksConfig` in `lit_trainer_callbacks.py`
- **Tests:** `tests/doc_classifier/test_custom_progress_bars.py`
- **Lightning Docs:** https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichProgressBar.html
