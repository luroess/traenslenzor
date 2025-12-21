from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DocClassifierResponse(BaseModel):
    """Structured response returned by the MCP tool."""

    probabilities: dict[str, float] = Field(
        description=("Mapping of class name to probability for the top-k predicted classes.")
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = ["DocClassifierResponse"]
