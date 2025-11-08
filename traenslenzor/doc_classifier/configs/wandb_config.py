from typing import Any

import wandb
from pydantic import Field
from pytorch_lightning.loggers import WandbLogger

from ..utils import BaseConfig
from .path_config import PathConfig


class WandbConfig(BaseConfig):
    """Thin wrapper around Lightning's WandbLogger."""

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

    def setup_target(self, **kwargs: Any) -> WandbLogger:
        """Instantiate a configured WandbLogger."""
        return WandbLogger(
            name=self.name,
            project=self.project,
            entity=self.entity,
            save_dir=PathConfig().wandb.as_posix(),
            offline=self.offline,
            log_model=self.log_model,
            prefix=self.prefix,
            experiment=wandb.run,
            checkpoint_name=self.checkpoint_name,
            tags=self.tags,
            group=self.group,
            job_type=self.job_type,
            **(kwargs or {}),
        )
