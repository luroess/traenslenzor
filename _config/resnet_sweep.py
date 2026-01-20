# resnet50_optuna_sweep.py
from copy import deepcopy

from pydantic import field_validator

from traenslenzor.doc_classifier.configs import ExperimentConfig, OptunaConfig
from traenslenzor.doc_classifier.data_handling.huggingface_rvl_cdip_ds import RVLCDIPConfig
from traenslenzor.doc_classifier.data_handling.transforms import (
    FineTunePlusTransformConfig,
    FineTuneTransformConfig,
    TransformConfig,
)
from traenslenzor.doc_classifier.lightning.lit_datamodule import DocDataModuleConfig
from traenslenzor.doc_classifier.lightning.lit_module import (
    BackboneType,
    DocClassifierConfig,
    OneCycleSchedulerConfig,
    OptimizerConfig,
)
from traenslenzor.doc_classifier.lightning.lit_trainer_callbacks import TrainerCallbacksConfig
from traenslenzor.doc_classifier.lightning.lit_trainer_factory import TrainerFactoryConfig
from traenslenzor.doc_classifier.utils import Optimizable, optimizable_field


class SweepOptimizerConfig(OptimizerConfig):
    weight_decay: float = optimizable_field(
        default=3e-4,
        optimizable=Optimizable.continuous(
            low=1e-6,
            high=1e-2,
            log=True,
        ),
    )
    backbone_lr_scale: float = optimizable_field(
        default=0.05,
        optimizable=Optimizable.continuous(
            low=0.01,
            high=0.2,
            log=True,
        ),
    )


class SweepSchedulerConfig(OneCycleSchedulerConfig):
    max_lr: float = optimizable_field(
        default=3e-5,
        optimizable=Optimizable.continuous(
            low=1e-4,
            high=3e-3,
            log=True,
        ),
    )
    pct_start: float = optimizable_field(
        default=0.12,
        optimizable=Optimizable.continuous(
            low=0.05,
            high=0.25,
        ),
    )

    # Fixed on purpose: these are typically second-order once max_lr is tuned.
    div_factor: float = 25
    final_div_factor: float = 10000


class SweepCallbacksConfig(TrainerCallbacksConfig):
    """Callback sweep parameters for ResNet-50 finetuning.

    Keep unfreeze early because:
    - 6 epochs + early stopping means late unfreezes often never happen.
    """

    use_optuna_pruning: bool = True
    backbone_unfreeze_at_epoch: int = optimizable_field(
        default=2,
        optimizable=Optimizable.discrete(
            low=2,
            high=3,
            step=1,
            description="Unfreeze backbone early enough that it actually gets trained even with early stopping.",
        ),
    )


TRANSFORM_SWEEP_CHOICES = (
    "train",
    "finetune",
    "finetune_plus",
)


class SweepDatasetConfig(RVLCDIPConfig):
    transform_config: TransformConfig = optimizable_field(
        default=FineTunePlusTransformConfig(),
        optimizable=Optimizable.categorical(
            choices=TRANSFORM_SWEEP_CHOICES,
            description="Choose data augmentation strategy for training.",
        ),
    )

    @field_validator("transform_config", mode="before")
    @classmethod
    def _resolve_transform_config(cls, value: object) -> TransformConfig | object:
        if isinstance(value, str):
            if value not in TRANSFORM_SWEEP_CHOICES:
                raise ValueError(f"Unknown transform_config choice '{value}'.")
            match value:
                case "train":
                    return deepcopy(TransformConfig())
                case "finetune":
                    return deepcopy(FineTuneTransformConfig())
                case "finetune_plus":
                    return deepcopy(FineTunePlusTransformConfig())
        return value


class ResNet50OptunaExperiment(ExperimentConfig):
    """To run this sweep, execute:
    ```bash
    uv run -m traenslenzor.doc_classifier.run --resnet_sweep
    ```
    """

    module_config: DocClassifierConfig = DocClassifierConfig(
        backbone=BackboneType.RESNET50,
        train_head_only=True,
        use_pretrained=True,
        optimizer=SweepOptimizerConfig(),
        scheduler=SweepSchedulerConfig(),
    )

    trainer_config: TrainerFactoryConfig = TrainerFactoryConfig(
        max_epochs=6,
        gradient_clip_val=1.0,
        callbacks=SweepCallbacksConfig(
            use_backbone_finetuning=True,
            use_early_stopping=False,
            early_stopping_patience=2,
        ),
    )

    datamodule_config: DocDataModuleConfig = DocDataModuleConfig(
        batch_size=64,
        limit_num_samples=0.3,  # speed for sweep
        train_ds=SweepDatasetConfig(),
    )

    optuna_config: OptunaConfig = OptunaConfig(
        n_trials=30,
        direction="minimize",
        monitor="val/loss",
        sampler="tpe",
        pruner="median",
    )
