from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocClassifierRequest(BaseModel):
    """Input payload for the ``classify_document`` MCP tool."""

    document_id: str = Field(
        description=(
            "Identifier returned by FileClient.upload/put; use it to fetch the image from the "
            "file server instead of supplying a filesystem path."
        )
    )
    top_k: int | str = Field(
        default=3,
        description="How many of the most probable classes to return.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("top_k", mode="before")
    @classmethod
    def _coerce_top_k(cls, v: int | str) -> int:
        """Accept int or digit string and coerce to bounded int."""
        if isinstance(v, str):
            if not v.isdigit():
                raise ValueError("top_k must be an integer or digit string")
            v = int(v)
        if not 1 <= v <= 16:
            raise ValueError("top_k must be between 1 and 16")
        return v


class DocClassifierResponse(BaseModel):
    """Structured response returned by the MCP tool."""

    probabilities: dict[str, float] = Field(
        description=("Mapping of class name to probability for the top-k predicted classes.")
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = ["DocClassifierRequest", "DocClassifierResponse"]
