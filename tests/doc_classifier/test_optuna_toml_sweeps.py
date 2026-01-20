import optuna

from traenslenzor.doc_classifier.configs.experiment_config import ExperimentConfig
from traenslenzor.doc_classifier.configs.optuna_config import OptunaConfig
from traenslenzor.doc_classifier.data_handling.transforms import FineTunePlusTransformConfig


def test_optuna_search_space_applies_suggestions_by_path() -> None:
    config = ExperimentConfig(
        optuna_config=OptunaConfig(
            search_space={
                "module_config": {
                    "optimizer": {
                        "backbone_lr_scale": {"low": 0.01, "high": 0.2, "log": True},
                    },
                },
                "trainer_config": {
                    "callbacks": {
                        "backbone_unfreeze_at_epoch": {"low": 0, "high": 3, "step": 1},
                    },
                },
            },
        ),
    )

    trial = optuna.trial.FixedTrial(
        {
            "module_config.optimizer.backbone_lr_scale": 0.05,
            "trainer_config.callbacks.backbone_unfreeze_at_epoch": 2,
        }
    )

    assert config.optuna_config is not None
    config.optuna_config.setup_optimizables(config, trial)

    assert config.module_config.optimizer.backbone_lr_scale == 0.05
    assert config.trainer_config.callbacks.backbone_unfreeze_at_epoch == 2
    assert (
        config.optuna_config.suggested_params["module_config.optimizer.backbone_lr_scale"] == 0.05
    )
    assert (
        config.optuna_config.suggested_params["trainer_config.callbacks.backbone_unfreeze_at_epoch"]
        == 2
    )


def test_optuna_search_space_can_swap_transform_config() -> None:
    config = ExperimentConfig(
        optuna_config=OptunaConfig(
            search_space={
                "datamodule_config": {
                    "train_ds": {
                        "transform_config": {
                            "choices": ["train", "finetune_plus"],
                        },
                    },
                },
            },
        ),
    )

    trial = optuna.trial.FixedTrial(
        {
            "datamodule_config.train_ds.transform_config": "finetune_plus",
        }
    )

    assert config.optuna_config is not None
    config.optuna_config.setup_optimizables(config, trial)

    transform_config = config.datamodule_config.train_ds.transform_config
    assert isinstance(transform_config, FineTunePlusTransformConfig)
