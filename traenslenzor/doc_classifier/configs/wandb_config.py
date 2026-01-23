import os
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field
from pytorch_lightning.loggers import WandbLogger

from ..utils import BaseConfig, Console
from .path_config import PathConfig

if TYPE_CHECKING:
    import pandas as pd


class WandbConfig(BaseConfig):
    """Wrapper around Lightning's [WandbLogger](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html)."""

    @property
    def target(self) -> type[WandbLogger]:
        return WandbLogger

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
    run_id: str | None = Field(
        default=None,
        description="Existing W&B run id to resume (e.g. 'dy83swag').",
    )
    resume: Literal["allow", "must", "never", "auto"] | None = Field(
        default=None,
        description="W&B resume policy (passed to wandb.init).",
    )
    resume_run_name: str | None = Field(
        default=None,
        description="Resume from a run display name (resolved via W&B API).",
    )
    resume_run_path: str | None = Field(
        default=None,
        description="Resume from a run path or URL (entity/project/run_id or wandb.ai URL).",
    )
    resume_checkpoint: bool = Field(
        default=True,
        description="When resuming, download a checkpoint artifact to restore model weights.",
    )
    resume_checkpoint_alias: str | None = Field(
        default="latest",
        description="Artifact alias to download when resuming (e.g. 'latest' or 'best').",
    )
    resume_checkpoint_artifact: str | None = Field(
        default=None,
        description="Explicit artifact name to download (optional).",
    )

    def has_resume_target(self) -> bool:
        return bool(self.run_id or self.resume_run_name or self.resume_run_path)

    @staticmethod
    def _coerce_datetime(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @staticmethod
    def _normalize_run_path(value: str) -> str:
        cleaned = value.strip().strip()
        cleaned = cleaned.split("?", 1)[0]
        if "wandb.ai" in cleaned:
            cleaned = cleaned.split("wandb.ai/", 1)[-1]
        cleaned = cleaned.strip().strip("/")
        cleaned = cleaned.replace("/runs/", "/")
        return cleaned

    def _parse_run_path(self, value: str) -> tuple[str | None, str | None, str]:
        cleaned = self._normalize_run_path(value)
        parts = [part for part in cleaned.split("/") if part]
        if len(parts) == 3:
            entity, project, run_id = parts
            return entity, project, run_id
        if len(parts) == 2:
            project, run_id = parts
            return None, project, run_id
        raise ValueError(
            "resume_run_path must be in the form 'entity/project/run_id', "
            "'project/run_id', or a wandb.ai URL."
        )

    @staticmethod
    def _flatten_dict(
        data: Mapping[str, Any],
        *,
        parent_key: str = "",
        sep: str = ".",
    ) -> dict[str, Any]:
        items: dict[str, Any] = {}
        for key, value in data.items():
            key_str = str(key)
            new_key = f"{parent_key}{sep}{key_str}" if parent_key else key_str
            if isinstance(value, Mapping):
                items.update(WandbConfig._flatten_dict(value, parent_key=new_key, sep=sep))
            else:
                items[new_key] = value
        return items

    def resolve_resume_run(self) -> tuple[str | None, Any | None]:
        """Resolve the run id (and optionally the run object) when resuming."""
        if self.run_id:
            return self.run_id, None

        if not self.has_resume_target():
            return None, None

        import wandb

        api = wandb.Api()
        if self.resume_run_path:
            entity, project, run_id = self._parse_run_path(self.resume_run_path)
            if self.entity and entity and self.entity != entity:
                raise ValueError(f"W&B entity mismatch: config '{self.entity}' vs path '{entity}'.")
            if self.project and project and self.project != project:
                raise ValueError(
                    f"W&B project mismatch: config '{self.project}' vs path '{project}'."
                )
            resolved_entity = self.entity or entity or getattr(api, "default_entity", None)
            if resolved_entity is None:
                raise ValueError("W&B entity is required to resolve resume_run_path.")
            resolved_project = project or self.project
            run = api.run(f"{resolved_entity}/{resolved_project}/{run_id}")
            self.run_id = run.id
            return run.id, run

        if not self.resume_run_name:
            return None, None

        resolved_entity = (
            self.entity or getattr(api, "default_entity", None) or os.getenv("WANDB_ENTITY")
        )
        if resolved_entity is None:
            raise ValueError("W&B entity is required to resolve resume_run_name.")
        resolved_project = self.project
        runs = list(api.runs(f"{resolved_entity}/{resolved_project}"))
        matches = [run for run in runs if run.name == self.resume_run_name]
        if not matches:
            raise ValueError(
                f"No W&B runs found for name '{self.resume_run_name}' in "
                f"{resolved_entity}/{resolved_project}."
            )
        matches.sort(
            key=lambda run: self._coerce_datetime(getattr(run, "updated_at", None))
            or self._coerce_datetime(getattr(run, "created_at", None))
            or datetime.min,
            reverse=True,
        )
        run = matches[0]
        self.run_id = run.id
        return run.id, run

    def build_runs_dataframe(
        self,
        runs: Iterable[Any],
        *,
        include_summary: bool = True,
        include_config: bool = True,
        include_unprefixed: bool = True,
        include_raw: bool = True,
        prefer: Literal["summary", "config"] = "summary",
        summary_prefix: str = "summary.",
        config_prefix: str = "config.",
        flatten_sep: str = ".",
    ) -> "pd.DataFrame":
        """Build a DataFrame from a list of W&B runs.

        Args:
            runs: Iterable of W&B run objects.
            include_summary: Include flattened summary metrics columns.
            include_config: Include flattened config columns.
            include_unprefixed: Include unprefixed merged columns for easy `query()`.
            include_raw: Include raw `summary` and `config` dict columns.
            prefer: Which source wins when summary/config share a key in unprefixed columns.
            summary_prefix: Prefix applied to flattened summary columns.
            config_prefix: Prefix applied to flattened config columns.
            flatten_sep: Separator used when flattening nested dicts.

        Returns:
            A pandas DataFrame with run metadata and flattened metrics/configs.
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for run in runs:
            summary_dict = getattr(getattr(run, "summary", None), "_json_dict", None) or {}
            if not isinstance(summary_dict, Mapping):
                summary_dict = {}
            config_dict = getattr(run, "config", None) or {}
            if not isinstance(config_dict, Mapping):
                config_dict = {}
            config_dict = {
                str(key): value
                for key, value in config_dict.items()
                if not str(key).startswith("_")
            }

            summary_flat = (
                self._flatten_dict(summary_dict, sep=flatten_sep) if include_summary else {}
            )
            config_flat = self._flatten_dict(config_dict, sep=flatten_sep) if include_config else {}

            match prefer:
                case "summary":
                    merged = dict(config_flat)
                    merged.update(summary_flat)
                case "config":
                    merged = dict(summary_flat)
                    merged.update(config_flat)
                case _:
                    raise ValueError("prefer must be 'summary' or 'config'.")

            row: dict[str, Any] = {
                "run_id": getattr(run, "id", None),
                "name": getattr(run, "name", None),
                "state": getattr(run, "state", None),
                "entity": getattr(run, "entity", None),
                "project": getattr(run, "project", None),
                "created_at": getattr(run, "created_at", None),
                "updated_at": getattr(run, "updated_at", None),
                "url": getattr(run, "url", None),
                "group": getattr(run, "group", None),
                "job_type": getattr(run, "job_type", None),
                "tags": list(getattr(run, "tags", []) or []),
            }

            if include_raw:
                row["summary"] = summary_dict
                row["config"] = config_dict

            if include_summary:
                row.update({f"{summary_prefix}{key}": value for key, value in summary_flat.items()})

            if include_config:
                row.update({f"{config_prefix}{key}": value for key, value in config_flat.items()})

            if include_unprefixed:
                for key, value in merged.items():
                    if key in row:
                        continue
                    row[key] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def runs_dataframe(
        self,
        *,
        entity: str | None = None,
        project: str | None = None,
        runs: Iterable[Any] | None = None,
        query: str | None = None,
        query_engine: Literal["python", "numexpr"] = "python",
        run_filters: dict[str, Any] | None = None,
        include_summary: bool = True,
        include_config: bool = True,
        include_unprefixed: bool = True,
        include_raw: bool = True,
        prefer: Literal["summary", "config"] = "summary",
        summary_prefix: str = "summary.",
        config_prefix: str = "config.",
        flatten_sep: str = ".",
    ) -> "pd.DataFrame":
        """Fetch W&B runs and return a DataFrame (optionally filtered by query).

        Args:
            entity: W&B entity override. Defaults to config/env/default entity.
            project: W&B project override. Defaults to config project.
            runs: Optional iterable of runs (bypass API call).
            query: Pandas query string applied to the resulting DataFrame.
            query_engine: Pandas query engine ("python" or "numexpr").
            run_filters: Optional W&B API filters dict.
            include_summary: Include flattened summary metrics columns.
            include_config: Include flattened config columns.
            include_unprefixed: Include unprefixed merged columns for easy `query()`.
            include_raw: Include raw `summary` and `config` dict columns.
            prefer: Which source wins when summary/config share a key in unprefixed columns.
            summary_prefix: Prefix applied to flattened summary columns.
            config_prefix: Prefix applied to flattened config columns.
            flatten_sep: Separator used when flattening nested dicts.

        Returns:
            A pandas DataFrame of run metadata and metrics.
        """
        if runs is None:
            import wandb

            api = wandb.Api()
            resolved_entity = (
                entity
                or self.entity
                or getattr(api, "default_entity", None)
                or os.getenv("WANDB_ENTITY")
            )
            resolved_project = project or self.project
            if resolved_entity is None:
                raise ValueError("W&B entity is required to list runs.")
            if resolved_project is None:
                raise ValueError("W&B project is required to list runs.")
            run_path = f"{resolved_entity}/{resolved_project}"
            if run_filters is None:
                runs = api.runs(run_path)
            else:
                runs = api.runs(run_path, filters=run_filters)

        df = self.build_runs_dataframe(
            runs,
            include_summary=include_summary,
            include_config=include_config,
            include_unprefixed=include_unprefixed,
            include_raw=include_raw,
            prefer=prefer,
            summary_prefix=summary_prefix,
            config_prefix=config_prefix,
            flatten_sep=flatten_sep,
        )

        if query:
            df = df.query(query, engine=query_engine)

        return df

    def download_resume_checkpoint(
        self,
        *,
        run: Any | None = None,
        root_dir: Path | None = None,
    ) -> Path | None:
        """Download a checkpoint artifact for a resumed run (if available)."""
        if not self.resume_checkpoint:
            return None

        console = Console.with_prefix(self.__class__.__name__, "resume_checkpoint")
        run_id, resolved_run = self.resolve_resume_run()
        run = run or resolved_run
        if run is None or run_id is None:
            console.warn("No run available to resume checkpoint from.")
            return None

        artifact = None
        if self.resume_checkpoint_artifact:
            artifact_ref = self.resume_checkpoint_artifact
            if ":" not in artifact_ref and self.resume_checkpoint_alias:
                artifact_ref = f"{artifact_ref}:{self.resume_checkpoint_alias}"
            try:
                artifact = run.use_artifact(artifact_ref)
            except Exception as exc:  # pragma: no cover - network dependent
                console.warn(f"Failed to resolve artifact '{artifact_ref}': {exc}")
                return None
        else:
            artifacts = [a for a in run.logged_artifacts() if getattr(a, "type", None) == "model"]
            if not artifacts:
                console.warn("No model artifacts logged for resumed run.")
                return None
            artifacts.sort(
                key=lambda art: self._coerce_datetime(getattr(art, "created_at", None))
                or datetime.min,
                reverse=True,
            )
            artifact = artifacts[0]
            if self.resume_checkpoint_alias:
                base_name = str(artifact.name).split(":", 1)[0]
                alias_ref = f"{base_name}:{self.resume_checkpoint_alias}"
                try:
                    artifact = run.use_artifact(alias_ref)
                except Exception:  # pragma: no cover - alias might not exist
                    console.warn(f"Alias '{alias_ref}' not found; using latest model artifact.")

        if artifact is None:
            console.warn("No artifact available to download.")
            return None

        download_root = Path(root_dir) if root_dir is not None else PathConfig().wandb
        download_root = download_root / "resume" / run_id
        download_root.mkdir(parents=True, exist_ok=True)
        try:
            artifact_dir = Path(artifact.download(root=download_root))
        except Exception as exc:  # pragma: no cover - network dependent
            console.warn(f"Failed to download artifact: {exc}")
            return None

        ckpt_candidates = sorted(
            artifact_dir.rglob("*.ckpt"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        if not ckpt_candidates:
            console.warn(f"No .ckpt files found in downloaded artifact at {artifact_dir}.")
            return None
        return ckpt_candidates[0]

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

        run_id, run = self.resolve_resume_run()
        resume_policy = self.resume or ("allow" if run_id else None)
        resolved_entity = self.entity
        resolved_project = self.project
        if run is not None:
            run_entity = getattr(run, "entity", None)
            run_project = getattr(run, "project", None)
            if self.entity and run_entity and self.entity != run_entity:
                raise ValueError(
                    f"W&B entity mismatch: config '{self.entity}' vs run '{run_entity}'."
                )
            if self.project and run_project and self.project != run_project:
                raise ValueError(
                    f"W&B project mismatch: config '{self.project}' vs run '{run_project}'."
                )
            resolved_entity = resolved_entity or run_entity
            resolved_project = resolved_project or run_project

        if resume_policy:
            kwargs = dict(kwargs or {})
            kwargs.setdefault("resume", resume_policy)

        return WandbLogger(
            name=self.name,
            project=resolved_project,
            entity=resolved_entity,
            save_dir=wandb_dir,
            offline=self.offline,
            log_model=self.log_model,
            prefix=self.prefix,
            # DO NOT pass experiment=wandb.run - this causes issues
            checkpoint_name=self.checkpoint_name,
            tags=self.tags,
            group=self.group,
            job_type=self.job_type,
            id=run_id,
            **(kwargs or {}),
        )
