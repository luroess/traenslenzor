from typing import Any

from pydantic import Field
from pytorch_lightning.loggers import WandbLogger

from ..utils import BaseConfig
from .path_config import PathConfig


class WandbConfig(BaseConfig):
    """Wrapper around Lightning's [WandbLogger](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html)."""

    target: type[WandbLogger] = Field(default_factory=lambda: WandbLogger, exclude=True)

    name: str | None = Field(default=None, description="Display name for the run.")
    project: str = Field(default="doc-class-detector", description="W&B project name.")
    entity: str | None = None
    offline: bool = Field(False, description="Enable offline logging.")
    log_model: bool | str = Field(
        default="all",
        description="Forward Lightning checkpoints to W&B artefacts.",
    )
    checkpoint_name: str | None = Field(default=None, description="Checkpoint artefact name.")
    tags: list[str] | None = Field(default=None, description="Optional list of tags.")
    group: str | None = Field(default=None, description="Group multiple related runs.")
    job_type: str | None = Field(default=None, description="Attach a W&B job_type label.")
    prefix: str | None = Field(default=None, description="Namespace prefix for metric keys.")

    def setup_target(self, **kwargs: Any) -> WandbLogger:
        """Instantiate a configured WandbLogger."""
        # Get wandb directory from centralized PathConfig
        wandb_dir = PathConfig().wandb.as_posix()

        # Initialize WandB with console logging disabled to prevent progress bar spam
        # The 'experiment' parameter should NOT be passed - let WandbLogger create the run
        # wandb.init(
        #     project=self.project,
        #     name=self.name,
        #     entity=self.entity,
        #     dir=wandb_dir,
        #     tags=self.tags,
        #     group=self.group,
        #     job_type=self.job_type,
        #     mode="offline" if self.offline else "online",
        #     # Disable console logging to prevent Rich progress bar updates from being logged
        #     settings=wandb.Settings(console="off"),
        #     reinit=True,  # Allow multiple wandb.init() calls
        # )

        return WandbLogger(
            name=self.name,
            project=self.project,
            entity=self.entity,
            save_dir=wandb_dir,
            offline=self.offline,
            log_model=self.log_model,
            prefix=self.prefix,
            # DO NOT pass experiment=wandb.run - this causes issues
            checkpoint_name=self.checkpoint_name,
            tags=self.tags,
            group=self.group,
            job_type=self.job_type,
            **(kwargs or {}),
        )
