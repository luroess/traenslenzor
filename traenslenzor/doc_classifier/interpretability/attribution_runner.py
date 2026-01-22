"""Utilities to run attribution scans and save plots to disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import LayerAttribution
from captum.attr import visualization as viz
from pydantic import Field

from ..data_handling.transforms import TransformConfig
from ..utils import BaseConfig, Console, Stage
from .attribution import (
    AttributionMethod,
    AttributionSample,
    BaselineStrategy,
    InterpretabilityConfig,
    find_best_worst_samples,
)

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer

    from ..configs.experiment_config import ExperimentConfig


@dataclass(slots=True)
class AttributionArtifacts:
    """Container for attribution plot artifacts."""

    image: np.ndarray
    attr_map: np.ndarray
    overlay: np.ndarray


class AttributionRunConfig(BaseConfig["AttributionRunner"]):
    """Configuration for running attribution scans and saving plots."""

    @property
    def target(self) -> type["AttributionRunner"]:
        return AttributionRunner

    enabled: bool = False
    """Whether to run attribution scan when executing the experiment."""

    num_samples: int = 1
    """Number of best/worst samples to attribute per model."""

    batch_size: int = 128
    """Batch size used for the best/worst scan."""

    num_workers: int = 16
    """DataLoader workers used for the best/worst scan."""

    methods: list[AttributionMethod] | None = None
    """Attribution methods to compute. Defaults to the supported fast methods."""

    output_root: Path | None = None
    """Override root directory for saved attribution plots."""

    mirror_test_split: bool = True
    """When True, reuses the test config for train/val to avoid heavy dataset loads."""

    force_normalization_mode: Literal["dataset", "imagenet"] | None = None
    """Override normalization_mode on transforms (e.g. 'dataset' or 'imagenet')."""

    force_grayscale_mean: float | None = None
    """Override grayscale mean for dataset normalization."""

    force_grayscale_std: float | None = None
    """Override grayscale std for dataset normalization."""

    force_convert_to_rgb: bool | None = None
    """Override convert_to_rgb on transforms when set."""

    interpretability: InterpretabilityConfig = Field(
        default_factory=lambda: InterpretabilityConfig(
            use_abs=True,
            n_steps=16,
            noise_samples=4,
            occlusion_window=(64, 64),
            occlusion_stride=(64, 64),
        ),
    )
    """Base interpretability config reused across methods."""

    baseline: BaselineStrategy = BaselineStrategy.ZERO
    """Baseline strategy used for attribution methods."""


class AttributionRunner:
    """Run attribution scans and persist plots to disk."""

    def __init__(self, config: AttributionRunConfig) -> None:
        self.config = config
        self.console = Console.with_prefix(self.__class__.__name__, "run")
        self._disabled_methods = {
            AttributionMethod.DEEP_LIFT,
            AttributionMethod.OCCLUSION,
            AttributionMethod.FEATURE_ABLATION,
        }

    def run(self, experiment: "ExperimentConfig") -> "Trainer":
        config = experiment.model_copy(deep=True)
        paths = config.paths

        self._apply_overrides(config)

        trainer, model, datamodule = config.setup_target(setup_stage=Stage.TEST)
        device = next(model.parameters()).device

        self._prepare_model(model)

        dataset = datamodule.test_ds
        transform_cfg = config.datamodule_config.test_ds.transform_config
        if transform_cfg is None:
            raise RuntimeError("Transform config missing for test split.")

        best, worst, class_names = find_best_worst_samples(
            model,
            dataset,
            device=device,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_samples=self.config.num_samples,
        )

        methods = (
            list(self.config.methods)
            if self.config.methods is not None
            else [
                AttributionMethod.GRAD_CAM,
                AttributionMethod.INTEGRATED_GRADIENTS,
                AttributionMethod.INPUT_X_GRADIENT,
                AttributionMethod.LAYER_GRAD_X_ACT,
                AttributionMethod.NOISE_TUNNEL_IG,
            ]
        )
        if self._disabled_methods:
            before = set(methods)
            methods = [m for m in methods if m not in self._disabled_methods]
            dropped = before.difference(methods)
            if dropped:
                self.console.warn(
                    f"Skipping disabled attribution methods: "
                    f"{', '.join(sorted(m.value for m in dropped))}"
                )

        output_root = (
            self.config.output_root
            if self.config.output_root is not None
            else paths.data_root / "attributions"
        )

        model_name = config.module_config.backbone.value
        for group_name, samples in [("best", best), ("worst", worst)]:
            for idx, sample in enumerate(samples, start=1):
                sample_dir = output_root / model_name / group_name / f"sample_{idx}"
                self._process_sample(
                    model=model,
                    transform_cfg=transform_cfg,
                    sample=sample,
                    class_names=class_names,
                    methods=methods,
                    output_dir=sample_dir,
                    ckpt_path=config.from_ckpt,
                    group_name=group_name,
                    sample_index=idx,
                )

        return trainer

    def _apply_overrides(self, experiment: "ExperimentConfig") -> None:
        if self.config.mirror_test_split:
            experiment.datamodule_config.train_ds = experiment.datamodule_config.test_ds.model_copy(
                deep=True
            )
            experiment.datamodule_config.val_ds = experiment.datamodule_config.test_ds.model_copy(
                deep=True
            )

        for ds_cfg in (
            experiment.datamodule_config.train_ds,
            experiment.datamodule_config.val_ds,
            experiment.datamodule_config.test_ds,
        ):
            ds_cfg.split = Stage.TEST
            transform_cfg = getattr(ds_cfg, "transform_config", None)
            if transform_cfg is None:
                continue
            self._override_transform(transform_cfg)

    def _override_transform(self, transform_cfg: TransformConfig) -> None:
        if self.config.force_normalization_mode is not None:
            transform_cfg.normalization_mode = self.config.force_normalization_mode
        if self.config.force_grayscale_mean is not None:
            transform_cfg.grayscale_mean = self.config.force_grayscale_mean
        if self.config.force_grayscale_std is not None:
            transform_cfg.grayscale_std = self.config.force_grayscale_std
        if self.config.force_convert_to_rgb is not None:
            transform_cfg.convert_to_rgb = self.config.force_convert_to_rgb

    def _prepare_model(self, model: "LightningModule") -> None:
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU) and module.inplace:
                module.inplace = False

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg", force=True)

    def _process_sample(
        self,
        *,
        model: "LightningModule",
        transform_cfg: TransformConfig,
        sample: AttributionSample,
        class_names: list[str],
        methods: list[AttributionMethod],
        output_dir: Path,
        ckpt_path: Path | None,
        group_name: str,
        sample_index: int,
    ) -> None:
        image = sample.image.to(next(model.parameters()).device)
        image_denorm = transform_cfg.unnormalize_tensor(image)
        image_np = self._to_numpy_image(image_denorm)

        label = self._label_name(sample.label, class_names)
        pred = self._label_name(sample.pred, class_names)
        conf = sample.conf

        self._save_image(output_dir / "original.png", image_np)

        for method in methods:
            cfg = self.config.interpretability.model_copy(deep=True)
            cfg.method = method
            cfg.baseline = self.config.baseline
            engine = cfg.setup_target(model)
            result = engine.attribute(image.unsqueeze(0), target=sample.pred)
            attr_map = self._attr_to_numpy(result.raw_attribution, image)

            method_dir = output_dir / method.value
            method_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "label": sample.label,
                "label_name": label,
                "pred": sample.pred,
                "pred_name": pred,
                "confidence": conf,
                "ckpt": str(ckpt_path) if ckpt_path is not None else None,
                "group": group_name,
                "sample_index": sample_index,
                "method": method.value,
            }

            (method_dir / "meta.json").write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

            self._save_attr_maps(method, attr_map, image_np, method_dir)

    def _save_attr_maps(
        self,
        method: AttributionMethod,
        attr_map: np.ndarray,
        image_np: np.ndarray,
        output_dir: Path,
    ) -> None:
        attr_map_vis = np.nan_to_num(attr_map, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = float(np.max(np.abs(attr_map_vis)))
        if max_val > 0.0:
            attr_map_vis = np.clip(np.abs(attr_map_vis) / max_val, 0.0, 1.0)
        else:
            attr_map_vis = np.zeros_like(attr_map_vis)
        if attr_map_vis.ndim == 3:
            if attr_map_vis.shape[2] == 1:
                attr_map_vis = attr_map_vis[:, :, 0]
            elif attr_map_vis.shape[2] not in (3, 4):
                attr_map_vis = attr_map_vis.mean(axis=2)
        plt.imsave(output_dir / "attr_map.png", attr_map_vis, cmap="magma")

        sign = self._viz_sign_for_method(method)
        outlier = 10.0 if method == AttributionMethod.NOISE_TUNNEL_IG else 0.0

        self._save_captum_plot(
            output_dir / "blended_heatmap.png",
            attr_map,
            image_np,
            method="blended_heat_map",
            sign=sign,
            outlier=outlier,
        )
        self._save_captum_plot(
            output_dir / "heatmap.png",
            attr_map,
            image_np,
            method="heat_map",
            sign=sign,
            outlier=outlier,
        )

    def _save_captum_plot(
        self,
        path: Path,
        attr_map: np.ndarray,
        image_np: np.ndarray,
        *,
        method: str,
        sign: str,
        outlier: float,
    ) -> None:
        attr_map = np.nan_to_num(attr_map)
        scale = float(np.max(np.abs(attr_map)))
        if scale == 0.0:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(image_np)
            ax.imshow(attr_map.mean(axis=2), cmap="magma", alpha=0.6)
            ax.axis("off")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return

        try:
            fig, _ = viz.visualize_image_attr(
                attr_map,
                image_np,
                method=method,
                sign=sign,
                show_colorbar=True,
                outlier_perc=outlier,
                use_pyplot=False,
            )
        except TypeError:
            _ = viz.visualize_image_attr(
                attr_map,
                image_np,
                method=method,
                sign=sign,
                show_colorbar=True,
                outlier_perc=outlier,
            )
            fig = plt.gcf()

        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _to_numpy_image(image: torch.Tensor) -> np.ndarray:
        img = image.detach().cpu()
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img.permute(1, 2, 0).numpy()

    @staticmethod
    def _attr_to_numpy(raw_attr: torch.Tensor, input_image: torch.Tensor) -> np.ndarray:
        attr = raw_attr.detach()
        if attr.dim() == 4:
            attr = attr[0]
        if attr.dim() == 3 and attr.shape[1:] != input_image.shape[1:]:
            attr = LayerAttribution.interpolate(attr.unsqueeze(0), input_image.shape[1:])[0]
        if attr.dim() == 3:
            attr = attr.permute(1, 2, 0)
        elif attr.dim() == 2:
            attr = attr.unsqueeze(-1)
        attr_np = attr.cpu().numpy().astype(np.float32, copy=False)
        attr_np = np.atleast_3d(attr_np)
        if attr_np.shape[2] == 1:
            attr_np = np.repeat(attr_np, 3, axis=2)
        return attr_np

    @staticmethod
    def _viz_sign_for_method(method: AttributionMethod) -> str:
        match method:
            case AttributionMethod.INTEGRATED_GRADIENTS | AttributionMethod.DEEP_LIFT:
                return "all"
            case AttributionMethod.NOISE_TUNNEL_IG:
                return "absolute_value"
            case AttributionMethod.GRAD_CAM:
                return "positive"
            case _:
                return "absolute_value"

    @staticmethod
    def _label_name(idx: int, class_names: list[str]) -> str:
        if class_names and 0 <= idx < len(class_names):
            return class_names[idx]
        return f"class_{idx}"

    @staticmethod
    def _save_image(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.clip(image, 0.0, 1.0)
        plt.imsave(path, image)
