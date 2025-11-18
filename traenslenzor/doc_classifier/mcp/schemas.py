from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DocClassifierRequest(BaseModel):
    """Input payload for the ``classify_document`` MCP tool."""

    path: Path = Field(
        description="Path to the image file to classify (any format Pillow can read)."
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=16,
        description="How many of the most probable classes to return.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DocClassifierResponse(BaseModel):
    """Structured response returned by the MCP tool."""

    probabilities: dict[str, float] = Field(
        description=(
            "Mapping of class name to probability (softmax scores for the returned top-k; "
            "values may not sum to 1 if top_k < num_classes)."
        )
    )
    top_k: int = Field(description="Number of classes included in the response.")

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = ["DocClassifierRequest", "DocClassifierResponse"]
