# resnet50_optuna_sweep.py


from traenslenzor.doc_classifier.configs import ExperimentConfig, OptunaConfig
from traenslenzor.doc_classifier.data_handling.huggingface_rvl_cdip_ds import RVLCDIPConfig
from traenslenzor.doc_classifier.data_handling.transforms import (
    TrainTransformConfig,
    ValTransformConfig,
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
from traenslenzor.doc_classifier.models.alexnet import AlexNet, AlexNetParams
from traenslenzor.doc_classifier.utils import Optimizable, optimizable_field

__all__ = ["AlexNet"]


class SweepOptimizerConfig(OptimizerConfig):
    weight_decay: float = optimizable_field(
        default=1e-3,
        optimizable=Optimizable.continuous(
            low=2e-4,
            high=1.2e-3,
            log=True,
        ),
    )


class SweepAlexNetParams(AlexNetParams):
    num_classes: int = 16
    dropout_prob: float = optimizable_field(
        default=0.5,
        optimizable=Optimizable.continuous(
            low=0.2,
            high=0.6,
        ),
    )


class SweepSchedulerConfig(OneCycleSchedulerConfig):
    max_lr: float = optimizable_field(
        default=3e-4,
        optimizable=Optimizable.continuous(
            low=8e-5,
            high=4e-4,
            log=True,
        ),
    )
    pct_start: float = 0.3
    div_factor: float = 25
    final_div_factor: float = 10000


class SweepDatasetConfig(RVLCDIPConfig):
    transform_config: TrainTransformConfig = TrainTransformConfig(convert_to_rgb=False)


class AlexNetOptunaExperiment(ExperimentConfig):
    """To run this sweep, execute:
    ```bash
    uv run -m traenslenzor.doc_classifier.run --alexnet_sweep
    ```
    """

    module_config: DocClassifierConfig = DocClassifierConfig(
        backbone=BackboneType.ALEXNET,
        train_head_only=False,
        use_pretrained=False,
        optimizer=SweepOptimizerConfig(),
        scheduler=SweepSchedulerConfig(),
        model_params=SweepAlexNetParams(),
    )

    trainer_config: TrainerFactoryConfig = TrainerFactoryConfig(
        max_epochs=6,
        gradient_clip_val=1.0,
        callbacks=TrainerCallbacksConfig(
            use_backbone_finetuning=False,
            use_early_stopping=False,
        ),
    )

    datamodule_config: DocDataModuleConfig = DocDataModuleConfig(
        batch_size=44,
        limit_num_samples=0.3,  # speed for sweep
        train_ds=RVLCDIPConfig(
            num_workers=15, transform_config=TrainTransformConfig(convert_to_rgb=False)
        ),
        val_ds=RVLCDIPConfig(
            split="val", transform_config=ValTransformConfig(convert_to_rgb=False)
        ),
    )

    optuna_config: OptunaConfig = OptunaConfig(
        study_name="doc-classifier-alexnet-sweep",
        n_trials=20,
        direction="minimize",
        monitor="val/loss",
        sampler="tpe",
        pruner="median",
    )
