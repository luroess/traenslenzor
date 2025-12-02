"""Runtime adapter used by the doc-classifier MCP server."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from fastmcp.exceptions import ToolError
from PIL import Image

from traenslenzor.doc_classifier.data_handling.transforms import ValTransformConfig
from traenslenzor.doc_classifier.lightning import DocClassifierConfig, DocClassifierModule
from traenslenzor.doc_classifier.utils import Console
from traenslenzor.file_server.client import FileClient

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

    async def classify_file_id(self, file_id: str, *, top_k: int) -> dict:
        """Classify an image referenced by file id (or local path) using the FileClient."""

        self.console.log(f"classify_file_id: received id='{file_id}' top_k={top_k}")
        try:
            if (image := await FileClient.get_image(file_id)) is None:
                raise ToolError(f"File id not found: {file_id}")

            if self._model is not None:
                self.console.log("classify_file_id: using loaded model for prediction")
                predictions = self._predict_with_model(image, top_k=top_k)
            else:
                if (raw_bytes := await FileClient.get_raw_bytes(file_id)) is None:
                    raise ToolError(f"File id not found: {file_id}")

                self.console.log("classify_file_id: using mock predictions (no checkpoint)")
                predictions = self._predict_mock_from_bytes(raw_bytes, top_k=top_k)

            self.console.dbg(
                f"classify_file_id: returning {len(predictions)} preds; "
                f"top label={predictions[0]['label']} p={predictions[0]['probability']:.3f}"
            )
            return {
                "predictions": predictions,
                "class_names": CLASS_NAMES,
            }
        except ToolError as te:
            self.console.error(f"classify_file_id: ToolError: {te}")
            raise te
        except Exception as exc:  # pragma: no cover - defensive guard
            self.console.error(f"classify_file_id: unexpected error: {exc}")
            raise ToolError(f"Unexpected classification failure: {exc}") from exc

    # --------------------------------------------------------- internal helpers
    def _load_model(self, checkpoint_path: Path) -> None:
        """Load a Lightning checkpoint, reconstructing params from hyper_parameters."""

        self.console.log(
            f"Loading checkpoint from {checkpoint_path} on device '{self.config.device}'."
        )

        ckpt = torch.load(
            checkpoint_path,
            map_location=self.config.device,
            weights_only=False,  # require full Lightning checkpoint (hyper_parameters + state_dict)
        )
        if not (isinstance(ckpt, dict) and "hyper_parameters" in ckpt):
            raise ToolError(
                "Invalid checkpoint format: expected Lightning checkpoint with 'hyper_parameters'."
            )

        try:
            params = DocClassifierConfig(**ckpt["hyper_parameters"])
        except Exception as exc:  # pragma: no cover
            raise ToolError(f"Failed to parse hyper_parameters: {exc}") from exc

        module = DocClassifierModule.load_from_checkpoint(
            checkpoint_path,
            params=params,
            strict=False,
            map_location=self.config.device,
        )
        module.eval()

        self._model = module
        self.model_version = checkpoint_path.stem
        self.console.log(
            f"Checkpoint loaded (version={self.model_version}, backbone={params.backbone})."
        )

    # TODO: move to lit module!
    def _predict_with_model(self, image: Image.Image, *, top_k: int) -> list[dict]:
        tensor = self._transform(image=np.array(image.convert("RGB")))["image"]
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

    def _predict_mock_from_bytes(self, raw_bytes: bytes, *, top_k: int) -> list[dict]:
        """Generate deterministic fake probabilities derived from raw file bytes."""
        scores = np.random.default_rng(self.config.seed).random(len(CLASS_NAMES))
        hashed = int.from_bytes(hashlib.sha256(raw_bytes).digest(), "big")
        scores[hashed % len(CLASS_NAMES)] += np.pi
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
