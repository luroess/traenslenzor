from enum import Enum
from typing import Generic, Optional, Sequence, Type, TypeVar, Union

import optuna
from pydantic import BaseModel, Field

T = TypeVar("T", int, float, bool, str, Enum)


class Optimizable(BaseModel, Generic[T]):
    target: Type[T]

    start: Optional[Union[int, float]] = Field(None)
    end: Optional[Union[int, float]] = Field(None)
    step: Optional[int] = Field(1)
    categories: Optional[Sequence[str]] = Field(None)
    log_scale: Optional[bool] = Field(False)

    def setup_target(self, name: str, trial: optuna.Trial) -> T:
        if self.target is int:
            if self.start is not None and self.end is not None:
                return trial.suggest_int(
                    name, low=self.start, high=self.end, step=self.step, log=self.log_scale
                )  # type: ignore
            else:
                raise ValueError("Integer target requires 'start' and 'end' values.")

        elif self.target is float:
            if self.start is not None and self.end is not None:
                return trial.suggest_float(name, low=self.start, high=self.end, log=self.log_scale)  # type: ignore
            else:
                raise ValueError("Float target requires 'start' and 'end' values.")

        elif self.target is bool:
            return trial.suggest_categorical(name, [True, False])  # type: ignore

        elif self.target is str:
            if self.categories is not None:
                return trial.suggest_categorical(name, self.categories)  # type: ignore
            else:
                raise ValueError("Categorical target requires 'categories' values.")
        elif issubclass(self.target, Enum):
            return trial.suggest_categorical(
                name, list(map(str, self.categories)) or list(map(str, self.target))
            )  # type: ignore

        else:
            raise ValueError(f"Unsupported or misconfigured target type: {self.target}")

    def as_field(self) -> Field:
        return Field(default_factory=lambda: self, validate_default=False)
