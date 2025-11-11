# Confusion Matrix Logging to WandB

## Overview

The `DocClassifierModule` automatically logs **confusion matrix plots and tables** to Weights & Biases at the end of each validation epoch. The confusion matrix shows how well the model distinguishes between different document classes, highlighting common misclassification patterns.

## Features

✅ **Automatic Logging** - Confusion matrix plot and table logged every validation epoch
✅ **Dual Format** - Both visual plot (heatmap) and interactive table for detailed analysis
✅ **Class Name Labels** - Uses actual class names from the dataset (e.g., "invoice", "letter")
✅ **WandB Integration** - Seamlessly integrates with WandB logger
✅ **Memory Efficient** - Figures are closed after logging to prevent memory leaks
✅ **Epoch Tracking** - Each confusion matrix is tagged with the epoch number

## How It Works

### 1. Confusion Matrix Metric

The module uses `torchmetrics.ConfusionMatrix` to accumulate predictions during validation:

```python
# In DocClassifierModule.__init__
self.confusion_matrix = torchmetrics.ConfusionMatrix(
    num_classes=self.params.num_classes,
    task="multiclass",
)

# In validation_step
self.confusion_matrix(logits, targets)
```

### 2. Class Name Extraction

At the end of each validation epoch, class names are extracted from the dataset:

```python
def _get_class_names(self) -> list[str] | None:
    """Get class names from the dataset."""
    try:
        # Access the datamodule's validation dataset
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            val_ds = self.trainer.datamodule.val_ds

            # Extract class names from HuggingFace dataset features
            if hasattr(val_ds, "features") and "label" in val_ds.features:
                label_feature = val_ds.features["label"]
                if hasattr(label_feature, "names"):
                    return label_feature.names
    except Exception as e:
        console.warn(f"Failed to get class names from dataset: {e}")

    return None
```

For RVL-CDIP, this returns:
```python
[
    'advertisement', 'budget', 'email', 'file folder', 'form',
    'handwritten', 'invoice', 'letter', 'memo', 'news article',
    'presentation', 'questionnaire', 'resume', 'scientific publication',
    'scientific report', 'specification'
]
```

### 3. Logging Architecture

The logging is cleanly separated into two dedicated methods:

#### `on_validation_epoch_end()`
Orchestrates the logging workflow:

```python
def on_validation_epoch_end(self) -> None:
    """Log confusion matrix plot and table to WandB at the end of validation epoch."""
    if self.logger is None:
        return

    # Compute confusion matrix
    confmat = self.confusion_matrix.compute()

    # Get class names from the datamodule
    class_names = self._get_class_names()

    # Log confusion matrix plot
    self._log_confusion_matrix_plot(confmat, class_names)

    # Log confusion matrix as table
    self._log_confusion_matrix_table(confmat, class_names)

    # Reset confusion matrix for next epoch
    self.confusion_matrix.reset()
```

#### `_log_confusion_matrix_plot()`
Creates and logs the visual heatmap:

```python
def _log_confusion_matrix_plot(
    self,
    confmat: Float[Tensor, "N N"],
    class_names: list[str] | None,
) -> None:
    """Create and log confusion matrix plot to WandB.

    Args:
        confmat (Tensor['N N', float]): Computed confusion matrix tensor.
        class_names: List of class names for axis labels, or None.
    """
    if not hasattr(self.logger, "experiment"):
        return

    import matplotlib.pyplot as plt
    import wandb

    # Create confusion matrix plot using torchmetrics
    fig, ax = self.confusion_matrix.plot(val=confmat)

    # Customize the plot with class names if available
    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, rotation=0, fontsize=8)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)
        ax.set_title(
            f"Confusion Matrix - Epoch {self.current_epoch}",
            fontsize=12,
            fontweight="bold",
        )

    # Log to WandB
    self.logger.experiment.log(
        {
            Metric.VAL_CONFUSION_MATRIX: wandb.Image(fig),
            "epoch": self.current_epoch,
        }
    )

    # Close the figure to free memory
    plt.close(fig)
```

#### `_log_confusion_matrix_table()`
Creates and logs the interactive WandB Table:

```python
def _log_confusion_matrix_table(
    self,
    confmat: Float[Tensor, "N N"],
    class_names: list[str] | None,
) -> None:
    """Log confusion matrix as WandB Table for interactive exploration.

    Args:
        confmat (Tensor['N N', float]): Computed confusion matrix tensor.
        class_names: List of class names for table labels, or None.
    """
    if not hasattr(self.logger, "experiment"):
        return

    import wandb

    # Convert tensor to numpy for table creation
    confmat_np = confmat.cpu().numpy()

    # Use class names if available, otherwise use indices
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(confmat_np))]

    # Create WandB Table with confusion matrix data
    # Columns: "Actual" + predicted class names
    columns = ["Actual"] + class_names
    data = []

    for i, actual_class in enumerate(class_names):
        row = [actual_class] + confmat_np[i].tolist()
        data.append(row)

    table = wandb.Table(columns=columns, data=data)

    # Log table to WandB
    self.logger.experiment.log(
        {
            f"{Metric.VAL_CONFUSION_MATRIX}_table": table,
            "epoch": self.current_epoch,
        }
    )
```

## Viewing in WandB

Once your training run is active, you can view the confusion matrix in two formats:

### 1. Confusion Matrix Plot (Heatmap)

1. **Navigate to your run** in the WandB dashboard
2. **Go to the "Media" tab** or search for `val/confusion_matrix`
3. **View confusion matrices** across epochs using the slider

**Confusion Matrix Interpretation:**
- **Rows**: True classes (ground truth)
- **Columns**: Predicted classes
- **Diagonal**: Correct predictions (darker = better)
- **Off-diagonal**: Misclassifications

**Example insights:**
- If row "invoice" has high values in column "form", the model confuses invoices with forms
- Dark diagonal indicates good per-class accuracy
- Light diagonal indicates poor per-class accuracy

### 2. Confusion Matrix Table (Interactive)

1. **Navigate to your run** in the WandB dashboard
2. **Go to the "Tables" section** or search for `val/confusion_matrix_table`
3. **Interact with the table**:
   - Sort by any column to find most confused classes
   - Filter rows to focus on specific classes
   - Export to CSV for further analysis

**Table Format:**

| Actual          | advertisement | budget | email | ... | specification |
|-----------------|---------------|--------|-------|-----|---------------|
| advertisement   | 145           | 2      | 1     | ... | 0             |
| budget          | 3             | 98     | 0     | ... | 1             |
| email           | 1             | 0      | 134   | ... | 0             |
| ...             | ...           | ...    | ...   | ... | ...           |
| specification   | 0             | 1      | 0     | ... | 89            |

**Benefits of the table format:**
- **Searchable**: Quickly find specific class confusions
- **Sortable**: Identify worst-performing class pairs
- **Exportable**: Download for offline analysis
- **Precise Values**: See exact counts without estimating from heatmap

## Customization

### Change Confusion Matrix Normalization

By default, the confusion matrix shows raw counts. To normalize:

```python
self.confusion_matrix = torchmetrics.ConfusionMatrix(
    num_classes=self.params.num_classes,
    task="multiclass",
    normalize="true",  # Options: None, "true", "pred", "all"
)
```

- `normalize="true"`: Normalize by true class (row-wise)
- `normalize="pred"`: Normalize by predicted class (column-wise)
- `normalize="all"`: Normalize by total samples

### Log Additional Metrics

You can extend `on_validation_epoch_end()` to log more visualizations:

```python
def on_validation_epoch_end(self) -> None:
    # ... existing confusion matrix code ...

    # Log per-class accuracy
    per_class_acc = confmat.diag() / confmat.sum(1)
    for i, acc in enumerate(per_class_acc):
        if class_names:
            self.log(f"val/acc_{class_names[i]}", acc)
        else:
            self.log(f"val/acc_class_{i}", acc)
```

### Custom Plot Styling

Modify the plot appearance:

```python
# After creating the plot
fig.set_size_inches(12, 10)  # Larger figure
ax.tick_params(labelsize=6)  # Smaller font for many classes

# Custom colormap
from matplotlib import pyplot as plt
import numpy as np

# Use a different colormap
ax.images[0].set_cmap('YlOrRd')  # Yellow-Orange-Red

# Add grid
ax.grid(False)  # Remove grid if present
```

## Troubleshooting

### Issue: Confusion Matrix Not Appearing in WandB

**Possible causes:**
1. WandB logger not configured (`use_wandb=false`)
2. Validation not running (check `limit_val_batches`)
3. WandB offline mode

**Solution:**
```toml
[trainer_config]
use_wandb = true
limit_val_batches = null  # Run full validation

[trainer_config.wandb_config]
offline = false
```

### Issue: Class Names Missing (Shows "Class 0", "Class 1", etc.)

**Cause:** Dataset features not accessible or missing `names` attribute

**Solution:** Verify dataset has class names:
```python
# In your code or notebook
dataset = config.setup_target()
print(dataset.features['label'].names)
```

### Issue: Out of Memory During Validation

**Cause:** Confusion matrix accumulating too many samples with large batch sizes

**Solution:** This is expected behavior - confusion matrix needs all samples. If memory is limited:
- Reduce batch size
- Use `limit_val_batches=0.5` to validate on 50% of data

## Metric Names

The confusion matrix is logged using standardized metric names:

```python
from traenslenzor.doc_classifier.utils.schemas import Metric

# Plot (heatmap visualization)
Metric.VAL_CONFUSION_MATRIX  # → "val/confusion_matrix"

# Table (interactive data)
f"{Metric.VAL_CONFUSION_MATRIX}_table"  # → "val/confusion_matrix_table"
```

Use these to query or filter in WandB dashboard.

## Integration with Other Tools

### Export Confusion Matrix Data

Access the raw confusion matrix for analysis:

```python
# At the end of validation epoch
confmat = self.confusion_matrix.compute()  # Tensor of shape (num_classes, num_classes)

# Convert to numpy
confmat_np = confmat.cpu().numpy()

# Save to CSV
import pandas as pd
df = pd.DataFrame(confmat_np, columns=class_names, index=class_names)
df.to_csv(f"confmat_epoch_{self.current_epoch}.csv")
```

### Log to Multiple Loggers

If using multiple loggers (e.g., TensorBoard + WandB):

```python
# In on_validation_epoch_end
for logger in self.loggers:  # self.loggers is a list
    if hasattr(logger, "experiment"):
        logger.experiment.log({
            Metric.VAL_CONFUSION_MATRIX: wandb.Image(fig),
            "epoch": self.current_epoch,
        })
```

## References

- **TorchMetrics ConfusionMatrix:** https://torchmetrics.readthedocs.io/en/stable/classification/confusion_matrix.html
- **TorchMetrics Plotting:** https://torchmetrics.readthedocs.io/en/stable/pages/plotting.html
- **WandB Media Logging:** https://docs.wandb.ai/guides/track/log/media
- **RVL-CDIP Dataset:** See `notebooks/rvl_cdip_dataset_overview.ipynb` for class name extraction examples
