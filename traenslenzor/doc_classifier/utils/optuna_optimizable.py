from enum import Enum
from typing import Any, Generic, Sequence, TypeVar

import optuna
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Optimizable(BaseModel, Generic[T]):
    """Declarative description of an optimisable parameter."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    target: type[Any] | None = Field(
        default=None,
        description="Python type of the parameter (int, float, bool, str, Enum).",
    )
    low: float | int | None = Field(default=None, description="Lower bound for numeric spaces.")
    high: float | int | None = Field(default=None, description="Upper bound for numeric spaces.")
    step: int | None = Field(default=1, description="Step used for discrete integer suggestions.")
    categories: Sequence[Any] | None = Field(
        default=None,
        description="Explicit set of categorical choices (or Enum members).",
    )
    log: bool = Field(default=False, description="Use logarithmic sampling for numeric spaces.")
    name: str | None = Field(
        default=None, description="Optional override for the Optuna parameter name."
    )
    default: T | None = Field(default=None, description="Default value used outside Optuna trials.")
    description: str | None = Field(
        default=None, description="Human readable description of the parameter."
    )

    @classmethod
    def continuous(
        cls,
        *,
        low: float,
        high: float,
        log: bool = False,
        name: str | None = None,
        default: float | None = None,
        description: str | None = None,
    ) -> "Optimizable[float]":
        return cls(
            target=float,
            low=low,
            high=high,
            log=log,
            name=name,
            default=default,
            description=description,
        )

    @classmethod
    def discrete(
        cls,
        *,
        low: int,
        high: int,
        step: int = 1,
        log: bool = False,
        name: str | None = None,
        default: int | None = None,
        description: str | None = None,
    ) -> "Optimizable[int]":
        return cls(
            target=int,
            low=low,
            high=high,
            step=step or 1,
            log=log,
            name=name,
            default=default,
            description=description,
        )

    @classmethod
    def categorical(
        cls,
        *,
        choices: Sequence[Any],
        name: str | None = None,
        default: Any | None = None,
        description: str | None = None,
    ) -> "Optimizable[Any]":
        return cls(
            categories=tuple(choices),
            name=name,
            default=default,
            description=description,
        )

    def suggest(self, trial: optuna.Trial, path: str) -> T:
        """Sample a value from Optuna."""
        name = self.name or path
        if self._is_categorical():
            choices = list(self._categorical_choices())
            value = trial.suggest_categorical(name, choices)
            return self._coerce(value)
        if self._is_bool():
            value = trial.suggest_categorical(name, [True, False])
            return self._coerce(value)
        if self._is_int():
            return self._coerce(
                trial.suggest_int(
                    name,
                    int(self._require_low()),
                    int(self._require_high()),
                    step=self.step or 1,
                    log=self.log,
                )
            )
        if self._is_float():
            return self._coerce(
                trial.suggest_float(
                    name,
                    float(self._require_low()),
                    float(self._require_high()),
                    log=self.log,
                )
            )
        raise ValueError(f"Unsupported optimizable configuration for '{path}'.")

    def serialize(self, value: Any) -> Any:
        """Convert a suggested value to a JSON/W&B friendly representation."""
        if isinstance(value, Enum):
            return value.value
        return value

    # ------------------------------------------------------------------ helpers
    def _is_bool(self) -> bool:
        return self.target is bool

    def _is_int(self) -> bool:
        if self.target is int:
            return True
        return (
            self.target is None
            and isinstance(self.low, int)
            and isinstance(self.high, int)
            and self.categories is None
        )

    def _is_float(self) -> bool:
        return self.target is float or (isinstance(self.low, float) or isinstance(self.high, float))

    def _is_categorical(self) -> bool:
        return self.categories is not None or (
            isinstance(self.target, type) and issubclass(self.target, Enum)
        )

    def _categorical_choices(self) -> Sequence[Any]:
        if self.categories is not None:
            return self.categories
        target = self.target
        if isinstance(target, type) and issubclass(target, Enum):
            return list(target)
        raise ValueError("Categorical optimizables require either categories or an Enum target.")

    def _require_low(self) -> float | int:
        if self.low is None:
            raise ValueError("Optimizable requires 'low'/'start'.")
        return self.low

    def _require_high(self) -> float | int:
        if self.high is None:
            raise ValueError("Optimizable requires 'high'/'end'.")
        return self.high

    def _coerce(self, value: Any) -> Any:
        target = self.target
        if target is None:
            return value
        if isinstance(target, type) and issubclass(target, Enum):
            if isinstance(value, target):
                return value
            return target(value)
        if target in {int, float, bool, str}:
            return target(value)
        return value


def optimizable_field(
    *,
    default: T,
    optimizable: Optimizable[T],
    **field_kwargs: Any,
) -> Any:
    """Attach an optimizable definition to a Field."""
    extras = dict(field_kwargs.pop("json_schema_extra", {}) or {})
    extras["optimizable"] = optimizable
    return Field(default=default, json_schema_extra=extras, **field_kwargs)
