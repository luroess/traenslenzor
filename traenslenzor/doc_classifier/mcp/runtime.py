"""Runtime adapter used by the doc-classifier MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from traenslenzor.doc_classifier.data_handling.transforms import ValTransformConfig
from traenslenzor.doc_classifier.lightning import (
    BackboneType,
    DocClassifierConfig,
    DocClassifierModule,
)
from traenslenzor.doc_classifier.utils import Console

if TYPE_CHECKING:
    from ..configs.mcp_config import DocClassifierMCPConfig

# RVL-CDIP class names
CLASS_NAMES: list[str] = [
    "advertisement",
    "budget",
    "email",
    "file folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news article",
    "presentation",
    "questionnaire",
    "resume",
    "scientific publication",
    "scientific report",
    "specification",
]


class DocClassifierRuntime:
    """Handles image preparation and (mock) inference for the MCP tool."""

    def __init__(self, config: DocClassifierMCPConfig) -> None:
        self.config = config
        self.model_version: str | None = None

        self.console = (
            Console.with_prefix(self.__class__.__name__, "init")
            .set_debug(self.config.is_debug)
            .set_verbose(self.config.verbose)
        )

        self._transform = ValTransformConfig(img_size=self.config.img_size).setup_target()
        self._model: DocClassifierModule | None = None
        if self.config.is_mock:
            self.console.log("Using mock model for document classification.")
        elif self.config.checkpoint_path is not None and Path(self.config.checkpoint_path).exists():
            try:
                self._load_model(Path(self.config.checkpoint_path))
            except Exception as exc:
                self.console.warn(
                    f"Failed to load checkpoint '{self.config.checkpoint_path}'. Falling back to mock: {exc}"
                )

    # ------------------------------------------------------------------ public
    def resolve_path(self, path: Path) -> Path:
        return Path(path).expanduser().resolve()

    def classify_path(self, path: Path, *, top_k: int) -> dict:
        path = self.resolve_path(path)
        if self._model is not None:
            predictions = self._predict_with_model(path, top_k=top_k)
        else:
            predictions = self._predict_mock(path, top_k=top_k)

        return {
            "predictions": predictions,
            "class_names": CLASS_NAMES,
        }

    # --------------------------------------------------------- internal helpers
    def _load_model(self, checkpoint_path: Path) -> None:
        """Lazy-load the Lightning module from a checkpoint when provided."""

        self.console.log(
            f"Loading checkpoint from {checkpoint_path} on device '{self.config.device}'."
        )

        # Keep the config minimal: inference only, no pretrained download.
        config = DocClassifierConfig(
            num_classes=len(CLASS_NAMES),
            backbone=BackboneType.ALEXNET,
            train_head_only=False,
            use_pretrained=False,
        )

        module = DocClassifierModule(config)
        state_dict = torch.load(checkpoint_path, map_location=self.config.device)

        # Support both Lightning-style and raw state dicts
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        module.load_state_dict(state_dict, strict=False)
        module.to(self.config.device)
        module.eval()

        self._model = module
        self.model_version = checkpoint_path.stem

    # TODO: move to lit module!
    def _predict_with_model(self, path: Path, *, top_k: int) -> list[dict]:
        image = Image.open(path).convert("RGB")
        tensor = self._transform(image=np.array(image))["image"]
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            logits = self._model.forward(tensor.to(self.config.device))
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        return [
            {
                "label": CLASS_NAMES[idx],
                "index": int(idx),
                "probability": float(probs[idx]),
            }
            for idx in top_indices
        ]

    def _predict_mock(self, path: Path, *, top_k: int) -> list[dict]:
        """Generate deterministic fake probabilities derived from the file bytes."""
        scores = np.random.default_rng(self.config.seed).random(len(CLASS_NAMES))
        # pick random winner based on file hash
        scores[hash(path.read_bytes()) % len(CLASS_NAMES)] += np.pi
        probs = np.exp(scores) / np.exp(scores).sum()

        top_indices = np.argsort(probs)[::-1][:top_k]
        return [
            {
                "label": CLASS_NAMES[idx],
                "index": int(idx),
                "probability": float(probs[idx]),
            }
            for idx in top_indices
        ]


__all__ = ["DocClassifierRuntime", "CLASS_NAMES"]
