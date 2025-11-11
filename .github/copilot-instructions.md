# trÄnslenzor

## Project Overview

**trÄnslenzor** is a modular and agentic document translation system aiming to capture or load a page image, translate text blocks, render a translated document, and classify the document type. Components act as MCP tools orchestrated by a local LLM supervisor. The system uses:
- **Supervisor** (LangGraph + local LLM) orchestrates multi-stage translation workflows with dynamic tool selection
- **MCP Tools** (FastMCP servers) expose isolated services: layout detection, font detection, document classification, file management
- **Document Classifier** **OUR RESPONSIBILITY**; (PyTorch Lightning) trains RVL-CDIP models with WandB tracking, Optuna sweeps, and Captum interpretability, uses **Config-as-Factory** pattern throughout—every runtime object instantiated via Pydantic `BaseConfig.setup_target()`; located in `traenslenzor/traenslenzor/doc_classifier`
- **Python Packaging** managed via `uv`; dependencies and project package installed via `uv sync` and `uv pip install -e .`; Python environment isolated in `.venv` (`/home/jandu/repos/traenslenzor/.venv/bin/python`); Use `uv run` to execute commands within this environment

## Style Guide

### Required Conventions

- ✓ **Typing**
    - All signatures must be typed; Use modern builtins (`list[str]`, `dict[str, Any]`)
    - Use `TYPE_CHECKING` guards for imports of types only used in annotations
    - Use `Literal` if applicable.
    - Use `jaxtyping` for tensor/array types with shape and dtype annotations.

- **Docstrings**: All public methods must have Google-style docstrings including type and shape for tensor/array arguments and return values. Keep descriptions concise but informative. Always include type and shape for tensors/arrays.

    ```python
    from jaxtyping import Float32
    from torch import Tensor

**Example (Typing + Docstring)**:

```python
def forward(
    self,
    test_tensor: Float32[Tensor, "B C H W"],
) -> tuple[
    Float32[Tensor, "B num_classes H W"],
    Float32[Tensor, "B num_classes"],
]:
"""Forward pass for Transformer.

Args:
    test_tensor (Tensor['B C H W', float32]): Batch of input images.

Returns:
    Tuple[Tensor, Tensor] containing:
        - Tensor['B num_classes H W', float32]: Output tensor after processing.
        - Tensor['B', float32]: Auxiliary output tensor.
"""
    ...
```

Further guidelines:
- ✓ Config classes inherit from a base config (e.g., `BaseConfig`)
- ✓ All functional classes (targets) and models instantiated via `setup_target()`
- ✓ Prefer dependency injection via config objects over hardcoded dependencies.
- ✓ Model selection via nested config union (`model_config: configA | configB`)
- ✓ Provide doc-strings for all relvant fields in pydantic classes or dataclasses, rather than using `Field(..., description="...")`. Don't use `Field(..., )` for primitive fields unless necessary (i.e, when `defaul_factory` is required). Example:
    ```py
    class MyConfig(BaseConfig):
        my_bool: bool = True
        """Whether to enable the awesome feature."""
    ```
- ✓ Separate setup and execution logic
- ✓ Prefer functional approaches over loops/comprehensions when appropriate
- ✓ Use `pathlib.Path` for all filesystem paths
- ✓ Work test-driven; every new feature must have corresponding tests in `tests/doc_classifier` using `pytest`
- ✓ Prefer `match-case` over `if-elif-else` for multi-branch logic when applicable
- ✓ Prefer `Enum` for categorical variables over string literals
- ✓ Use `Console` from `traenslenzor.doc_classifier.utils` for structured logging


## Architecture & Design Patterns

- **Package layout:** `traenslenzor/doc_classifier` is a uv subpackage with submodules `configs`, `data_handling`, `lightning`, `utils`.

- **Config-as-Factory:** Every runtime object is created via their Pydantic config's `setup_target()` method. Never instantiate runtime classes directly. Use `field_validator` and `model_validator` decorators for cleanly structured validation within configs.
    - Examples:
    ```python
    from pydantic import BaseModel, Field
    from traenslenzor.doc_classifier.lit_utils import BaseConfig

    class MyComponentConfig(BaseConfig["MyComponent"]):
        target: type["MyComponent"] = Field(default_factory=lambda: MyComponent, exclude=True)

        # Config fields
        learning_rate: float = 1e-3
        batch_size: int = 32

    class MyComponent:
        def __init__(self, config: MyComponentConfig):
            self.config = config
    ```
    - Config Composition
    ```python
    class ExperimentConfig(BaseConfig):
        trainer_config: TrainerFactoryConfig
        module_config: DocClassifierConfig
        datamodule_config: DocDataModuleConfig

        def setup_target(self) -> tuple[Trainer, LightningModule, LightningDataModule]:
            trainer = self.trainer_config.setup_target()
            module = self.module_config.setup_target()
            datamodule = self.datamodule_config.setup_target()
            return trainer, module, datamodule
    ```

- **Console Logging:** Use `Console` from `traenslenzor.doc_classifier.utils` for structured, context-aware logging:
    ```python
    from traenslenzor.doc_classifier.utils import Console

    console = Console.with_prefix(self.__class__.__name__, "setup_target")
    console.set_verbose(self.verbose).set_debug(self.is_debug)

    console.log("Starting setup...")          # Info when verbose=True
    console.warn("Deprecated parameter")       # Warning + caller stack
    console.error("Invalid configuration")     # Error + caller stack
    console.dbg("Internal state: ...")         # Debug when is_debug=True
    console.plog(complex_obj)                  # Pretty-print with devtools
    ```

- **Centralized Path Handling** via `PathConfig(SingletonConfig)` for global resources (paths, environment):
```python
# file: traenslenzor/doc_classifier/configs/path_config.py
from traenslenzor.doc_classifier.utils import SingletonConfig

class PathConfig(SingletonConfig):
    root: Path = Field(default_factory=lambda: Path(__file__).parents[3])
    data_root: Path = Field(default_factory=lambda: Path("data/rvl_cdip"))
```

- **Training Pipeline**
    1. **Config**: Define `ExperimentConfig` with nested `DocClassifierConfig`, `DocDataModuleConfig`, `TrainerFactoryConfig`
    2. **Optuna (optional)**: Call `experiment_config.run_optuna_study()` for hyperparameter sweeps
    3. **Run**: `experiment_config.setup_target_and_run(stage="train")`
    4. **Artifacts**: Checkpoints → `PathConfig().checkpoints`, WandB runs → `PathConfig().wandb`

- **Model Variants**
    - **AlexNet**: Custom implementation in `lightning/lit_module.py`
    - **ResNet-50**: torchvision pretrained, replace classification head, optionally fine-tune backbone
    - **Vision Transformer**: torchvision pretrained ViT, re-head for 16-class RVL-CDIP

    Select via config union:
    ```python
    class DocClassifierConfig(BaseConfig):
        model_config: AlexNetConfig | ResNet50Config | ViTConfig
    ```
- **Data Handling**
    - **Dataset**: `RVLCDIPDataset` reads from CSV annotations + images directory
    - **Transforms**: Albumentations for augmentation (color jitter, perspective, random crop), torchvision for final tensor conversion
    - **PathConfig**: Centralizes `data_root`, `images_dir`, `annotations_csv` to avoid hardcoded paths


## Context7 IDs

# Use these Context7 IDs with the `#get-library-docs` mcp tool:

- [`/pydantic/pydantic`](https://context7.com/websites/pydantic_dev)
- [`/lightning-ai/pytorch-lightning`](https://context7.com/lightning-ai/pytorch-lightning)
- [`/optuna/optuna`](https://context7.com/websites/optuna_readthedocs_io_en_stable)
- [`/wandb/wandb`](https://context7.com/websites/docs_wandb_ai)
- [`/pytorch/captum`](https://context7.com/pytorch/captum)
- [`/lightning-ai/torchmetrics`](https://context7.com/lightning-ai/torchmetrics)
- [`albumentations.ai/docs`](https://context7.com/websites/albumentations_ai)
- [`/huggingface/datasets`](https://context7.com/huggingface/datasets)
- [`/rocm/pytorch`](https://context7.com/rocm/pytorch)