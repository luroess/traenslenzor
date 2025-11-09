# Document Classifier

High-level package that trains and serves the RVL-CDIP document class detector.

## Setup Instructions

**Python Version:** Python 3.13.9

```bash
cd traenslenzor/

# Install dependencies into .venv
uv sync --group dev --group classifier --group notebook

# Install the project package in editable mode to allow importing the traenslenzor package
uv pip install -e .

# Verify installation
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Expected output: 2.9.0+cu128 12.8 True

# Verify traenslenzor package
uv run python -c "import traenslenzor; print('Package installed successfully!')"
```

**Important:** After `uv sync`, you must run `uv pip install -e .` to make the `traenslenzor` package importable. This is because `uv sync` only installs dependencies listed in `pyproject.toml`, not the project itself.


### WandB Setup

1. Create a [Weights & Biases](https://wandb.ai/site) account and get your API key if you don't have one.
2. Run `uv run wandb login` and paste your API key!

## Package Structure

- `configs/` – Pydantic configs (`ExperimentConfig`, `PathConfig`, `WandbConfig`, etc.) that don't have their own "target" / runtime object.
- `data_handling/` – RVL-CDIP dataset + transform factories.
- `lightning/` – LightningModule/DataModule/Trainer factory definitions.
- `utils/` – Shared rich `Console`, `BaseConfig` other utilities.

## Design Patterns

- **Config-as-Factory** – Each runtime object is created via `config.setup_target(*args, **kwargs)`. Nested configs compose larger systems (e.g., `ExperimentConfig` -> trainer/module/datamodule).
- **Singleton PathConfig** – Centralised filesystem wiring so datasets, checkpoints, and wandb logs stay consistent across modules.
- **Console Prefixing** – All user-facing logs go through `Console.with_prefix(...)` to keep CLI output structured.
- **Optuna-aware Fields** – Any config field can declare a search space using `optimizable_field(..., optimizable=Optimizable(...))`. During `ExperimentConfig.run_optuna_study()` the copied config is automatically mutated with values suggested by the current trial. Suggestions propagate to nested configs/lists/dicts and are mirrored in `OptunaConfig.suggested_params` -> logged to WandB.

### Optuna Integration Cheatsheet

```python
from traenslenzor.doc_classifier.optuna import Optimizable, optimizable_field
from traenslenzor.doc_classifier.configs import ExperimentConfig, OptunaConfig

class MyModuleConfig(DocClassifierConfig):
    # Search learning rate between 1e-4 and 5e-3
    learning_rate: float = optimizable_field(
        default=1e-3,
        optimizable=Optimizable.continuous(low=1e-4, high=5e-3),
    )

cfg = ExperimentConfig(
    module_config=MyModuleConfig(),
    optuna_config=OptunaConfig(n_trials=25, monitor="val/loss"),
)
cfg.run_optuna_study()
```

Key points:

1. Define `optimizable_field` (or set `json_schema_extra={"optimizable": Optimizable(...)}` manually) on any BaseConfig attribute. Supported spaces: `continuous`, `discrete`, categorical/enums, booleans.
2. `OptunaConfig.setup_optimizables()` recursively visits nested configs, lists, and dicts; suggested values are applied before `ExperimentConfig.setup_target(...)`.
3. Suggestions are recorded in `OptunaConfig.suggested_params` and forwarded to WandB via `log_to_wandb()`. No manual plumbing needed inside your objective.

## Running Experiments

### CLI Usage

The `run.py` script provides a simple CLI interface powered by Pydantic's automatic CLI parsing. You can run experiments by:

1. **Using a YAML config file:**

```bash
# Create or export a config file first
uv run traenslenzor/doc_classifier/export_demo_config.py

# Run training
uv run traenslenzor/doc_classifier/run.py --config_path=demo_experiment_config.yaml

# Run validation only
uv run traenslenzor/doc_classifier/run.py --config_path=demo_experiment_config.yaml --stage=VAL

# Run test only
uv run traenslenzor/doc_classifier/run.py --config_path=demo_experiment_config.yaml --stage=TEST
```

2. **Using command-line arguments (override defaults):**

```bash
# Quick dev run with overrides
uv run traenslenzor/doc_classifier/run.py \
  --is_debug=true \
  --verbose=true \
  --trainer_config.max_epochs=3 \
  --datamodule_config.batch_size=64 \
  --module_config.learning_rate=0.001

# Optuna hyperparameter search
uv run traenslenzor/doc_classifier/run.py \
  --run_optuna=true \
  --optuna_config.n_trials=20 \
  --optuna_config.monitor="val/loss"
```

3. **Combining both (CLI args override config file):**

```bash
uv run traenslenzor/doc_classifier/run.py \
  --config_path=demo_experiment_config.yaml \
  --trainer_config.max_epochs=10 \
  --module_config.learning_rate=0.0005
```

**Key CLI Features:**

- Automatic parsing via `pydantic-settings` with `cli_parse_args=True`
- Nested config access using dot notation (e.g., `--trainer_config.max_epochs=10`)
- Type validation and conversion handled by Pydantic
- Environment variable support (prefix with `EXPERIMENT_`)
- Clean error messages for invalid arguments

**Common Workflows:**

```bash
# Full training run with WandB logging
uv run traenslenzor/doc_classifier/run.py \
  --wandb_config.project_name="rvl-cdip-classification" \
  --wandb_config.run_name="resnet50-baseline"

# Debug run (fast_dev_run, no checkpoints)
uv run traenslenzor/doc_classifier/run.py \
  --is_debug=true \
  --trainer_config.fast_dev_run=true

# Resume from checkpoint
uv run traenslenzor/doc_classifier/run.py \
  --resume_from_checkpoint="checkpoints/last.ckpt"

# Hyperparameter sweep with Optuna
uv run traenslenzor/doc_classifier/run.py \
  --run_optuna=true \
  --optuna_config.n_trials=50 \
  --optuna_config.study_name="resnet50-lr-sweep"
```

## Resources

### RVL-CDIP Dataset

- [Hugging Face :: chainyo/rvl-cdip](https://huggingface.co/datasets/chainyo/rvl-cdip) classification dataset (16 classes)
- [Hugging Face :: RIPS-Goog-23/RVL-CDIP](https://huggingface.co/datasets/RIPS-Goog-23/RVL-CDIP) provides AABBs and OCR text
- [Kaggle](https://www.kaggle.com/datasets/pdavpoojan/the-rvlcdip-dataset-test)

### Context7 IDs

- [`/pydantic/pydantic`](https://context7.com/websites/pydantic_dev)
- [`/lightning-ai/pytorch-lightning`](https://context7.com/lightning-ai/pytorch-lightning)
- [`/optuna/optuna`](https://context7.com/websites/optuna_readthedocs_io_en_stable)
- [`/wandb/wandb`](https://context7.com/websites/docs_wandb_ai)
- [`/pytorch/captum`](https://context7.com/pytorch/captum)
- [`/lightning-ai/torchmetrics`](https://context7.com/lightning-ai/torchmetrics)
- [`albumentations.ai/docs`](https://context7.com/websites/albumentations_ai)
