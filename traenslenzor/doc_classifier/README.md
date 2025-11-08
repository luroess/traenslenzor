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