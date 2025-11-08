from enum import Enum

from datasets import Split
from typing_extensions import Self


class Stage(Enum):
    """
    (TRAIN, VAL, TEST) = ("train", "val", "test")
    """

    TRAIN = ("train", "fit")
    VAL = ("val", "validate")
    TEST = ("test",)

    def __str__(self):
        return self.value[0]

    @classmethod
    def from_str(cls, value: str | Self) -> Self:
        """Map strings (e.g. 'fit', 'validate') back to Stage members."""
        if isinstance(value, cls):
            return value
        for member in cls:
            if value in member.value:
                return member
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


__all__ = ["Stage"]
