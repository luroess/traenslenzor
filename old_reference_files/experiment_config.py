from copy import deepcopy
from datetime import datetime
from typing import Any, Optional, Tuple, Type

import optuna
import wandb
from litutils import CONSOLE, BaseConfig, OptunaConfig, PathConfig, Stage
from pydantic import Field, model_validator
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from typing_extensions import Self

from .lit_datamodule import DatamoduleParams, LitDataModule
from .lit_module import ImgClassifierParams, LitImageClassifierModule
from .lit_trainer_factory import TrainerConfig


class ExperimentConfig(BaseConfig):
    """
    TODO: add TrainerConfig
    """

    is_debug: bool = True
    verbose: bool = True
    run_name: str = Field(
        default_factory=lambda: datetime.now().strftime("R%Y-%m-%d_%H:%M:%S")
    )

    from_ckpt: Optional[str] = None

    is_fast_dev_run: bool = False
    paths: PathConfig = Field(default_factory=PathConfig)
    trainer_config: TrainerConfig = Field(default_factory=TrainerConfig)

    module_config: ImgClassifierParams = ImgClassifierParams()
    module_type: Type[LitImageClassifierModule] = Field(
        default_factory=lambda: LitImageClassifierModule
    )  # Expect a class derived from LightningModule
    datamodule_config: DatamoduleParams = DatamoduleParams()
    datamodule_type: Type[LitDataModule] = Field(
        default_factory=lambda: LitDataModule
    )  # Expect a class derived from LightningDataModule

    optuna_config: Optional[OptunaConfig] = None

    def setup_target(
        self, setup_stage: Optional[Stage] = Stage.TRAIN, **kwargs: Any
    ) -> Tuple[Trainer, LightningModule, LightningDataModule]:
        """Create trainer, module and datamodule instances.

        Returns:
            Tuple containing:
            - PyTorch Lightning Trainer
            - LightningModule instance
            - LightningDataModule instance
        """
        # Setup trainer first
        trainer = self.trainer_config.setup_target(self, **kwargs)

        # Setup module with checkpoint handling
        if self.from_ckpt:
            try:
                from_ckpt = self.paths.checkpoints / self.from_ckpt
                assert from_ckpt.exists()
                CONSOLE.log(f"Loading model from checkpoint: {from_ckpt}")
                lit_module = self.module_type.load_from_checkpoint(
                    checkpoint_path=from_ckpt, params=self.module_config
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint: {e}")
        else:
            lit_module = self.module_config.setup_target()

        # Setup datamodule
        lit_datamodule = self.datamodule_config.setup_target()
        if setup_stage:
            lit_datamodule.setup(stage=setup_stage)
            assert lit_datamodule.train_ds is not None
            assert lit_datamodule.train_ds.num_classes == lit_module.params.num_classes

        CONSOLE.log("Experiment setup complete!")
        return trainer, lit_module, lit_datamodule

    def run_optuna_study(self) -> None:
        """Integrate Optuna for hyperparameter optimization"""
        assert self.optuna_config is not None, "OptunaConfig is not set!"

        def objective(trial: optuna.Trial) -> float:
            # Create a deep copy of the experiment config to preserve original Optimizable fields
            experiment_config_copy = deepcopy(self)

            # Setup the experiment, and train
            experiment_config_copy.trainer_config.wandb_config.name = (
                f"{self.run_name}_T{trial.number}"
            )
            experiment_config_copy.optuna_config.setup_optimizables(experiment_config_copy, trial)  # type: ignore
            trainer, lit_module, lit_datamodule = experiment_config_copy.setup_target(
                trial=trial
            )
            trainer.fit(lit_module, datamodule=lit_datamodule)

            monitor = experiment_config_copy.optuna_config.monitor  # type: ignore
            metric = trainer.callback_metrics[monitor].item() or float("inf")
            CONSOLE.log(
                f"Trial {trial.number} completed with\n{monitor}: {metric},\nparams:\n{trial.params}"
            )
            experiment_config_copy.optuna_config.log_to_wandb()  # type: ignore

            # End current Wandb run
            wandb.finish()

            return metric

        # Create an Optuna study and optimize
        self.trainer_config.wandb_config.group = "optuna"
        self.trainer_config.wandb_config.job_type = f"Opt:{self.run_name}"
        self.optuna_config.setup_target().optimize(
            objective,
            n_trials=self.optuna_config.n_trials,
        )

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        self.trainer_config.update_wandb_config(self)

        return self
