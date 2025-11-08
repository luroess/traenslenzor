from enum import StrEnum

from datasets import Split
from typing_extensions import Self


class Stage(StrEnum):
    """Stages of the training lifecycle.

    Members:
        TRAIN: "train"
        VAL: "val"
        TEST: "test"
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str | Self) -> Self:
        """Map strings (e.g. 'fit', 'validate') back to Stage members."""
        if isinstance(value, cls):
            return value
        alias_map: dict[str, Stage] = {
            "fit": cls.TRAIN,
            "validate": cls.VAL,
        }
        if value in alias_map:
            return alias_map[value]
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Unknown stage value '{value}' of type {type(value)}.")

    def to_split(self) -> Split:
        """Convert Stage to Hugging Face Datasets Split."""
        match self:
            case Stage.TRAIN:
                return Split.TRAIN
            case Stage.VAL:
                return Split.VALIDATION
            case Stage.TEST:
                return Split.TEST


class Metric(StrEnum):
    """Standardized metric names for logging during training, validation, and testing.

    Format: {stage}/{metric_type}

    Examples:
        - Metric.TRAIN_LOSS -> "train/loss"
        - Metric.VAL_ACCURACY -> "val/accuracy"
    """

    # Training metrics
    TRAIN_LOSS = "train/loss"
    TRAIN_ACCURACY = "train/accuracy"

    # Validation metrics
    VAL_LOSS = "val/loss"
    VAL_ACCURACY = "val/accuracy"

    # Test metrics
    TEST_LOSS = "test/loss"
    TEST_ACCURACY = "test/accuracy"

    def __str__(self) -> str:
        return self.value


# Backwards compatibility alias for legacy imports
MetricName = Metric


__all__ = ["Stage", "Metric", "MetricName"]
