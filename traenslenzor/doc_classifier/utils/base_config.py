import tomllib
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    ForwardRef,
    Generic,
    Optional,
    Set,
    Type,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rich.text import Text
from rich.tree import Tree
from tomlkit import TOMLDocument, aot, array, comment, document, dumps, string, table
from tomlkit.items import Table

from .console import Console

TargetType = TypeVar("TargetType")


class NoTarget:
    @staticmethod
    def setup_target(config: "BaseConfig", **kwargs: Any) -> None:
        return None


class BaseConfig(BaseModel, Generic[TargetType]):
    target: Callable[["BaseConfig[TargetType]"], TargetType] = Field(
        default_factory=lambda: NoTarget,
        exclude=True,
        json_schema_extra={
            "type": "string",
            "description": "Internal callable target, not configurable via CLI/JSON",
        },
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    propagated_fields: Dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description="Tracks fields propagated from parent configs",
    )

    def setup_target(self, **kwargs: Any) -> TargetType:
        if not callable(factory := getattr(self.target, "setup_target", self.target)):
            Console().print(
                f"Target '[bold yellow]{self.target}[/bold yellow]' of type [bold yellow]{factory.__class__.__name__}[/bold yellow] is not callable."
            )
            raise ValueError(
                f"Target '{self.target}' of type {factory.__class__.__name__} is not callable / does not have a 'setup_target' or '__init__' method."
            )

        return factory(self, **kwargs)

    # --------------------------------------------------------------------- TOML IO
    def to_toml(
        self,
        path: Path | None = None,
        *,
        include_comments: bool = True,
        include_type_hints: bool = True,
    ) -> str:
        """Serialise the config (and nested configs) to TOML.

        Args:
            path: Optional path to write the TOML to.
            include_comments: When true, injects docstrings/descriptions as comments.
            include_type_hints: When true, append type information to the comments.

        Returns:
            The rendered TOML string.
        """
        doc = document()
        if include_comments and self.__class__.__doc__:
            for line in self.__class__.__doc__.strip().splitlines():
                if line.strip():
                    doc.add(comment(line.strip()))
        self._write_toml_fields(
            container=doc,
            include_comments=include_comments,
            include_type_hints=include_type_hints,
        )
        rendered = dumps(doc)
        if path is not None:
            Path(path).write_text(rendered, encoding="utf-8")
        return rendered

    def save_toml(
        self,
        path: Path | str,
        *,
        include_comments: bool = True,
        include_type_hints: bool = True,
    ) -> Path:
        """Persist the configuration to a TOML file and return the resolved path."""
        target_path = Path(path)
        self.to_toml(
            path=target_path,
            include_comments=include_comments,
            include_type_hints=include_type_hints,
        )
        return target_path

    @classmethod
    def from_toml(cls: Type["BaseConfig"], source: str | Path | bytes) -> "BaseConfig":
        """Load a config from a TOML string or file path."""
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, bytes):
            text = source.decode("utf-8")
        else:
            potential_path = Path(source)
            if potential_path.exists():
                text = potential_path.read_text(encoding="utf-8")
            else:
                text = source

        data = tomllib.loads(text)
        return cls.model_validate(data)

    # ------------------------------------------------------------------ Visualization
    def inspect(self, show_docs: bool = False) -> None:
        tree = self._build_tree(show_docs=show_docs, _seen_singletons=set())
        Console().print(tree, soft_wrap=False, highlight=True, markup=True, emoji=False)

    def _build_tree(  # pragma: no cover - visualization helper
        self,
        show_docs: bool = False,
        _seen_singletons: Optional[Set[int]] = None,
        _is_top_level: bool = True,
        _seen_path_configs: Optional[Set[int]] = None,
    ) -> Tree:
        if _seen_singletons is None:
            _seen_singletons = set()
        if _seen_path_configs is None:
            _seen_path_configs = set()

        tree = Tree(Text(self.__class__.__name__, style="config.name"))

        if show_docs and self.__class__.__doc__:
            tree.add(Text(self.__class__.__doc__, style="config.doc"))

        for field_name, field in self.__class__.model_fields.items():
            value = getattr(self, field_name)
            field_style = (
                "config.propagated" if field_name in self.propagated_fields else "config.field"
            )

            # Handle singleton configs (only once)
            if isinstance(value, SingletonConfig):
                # Check if it's a PathConfig
                is_path_config = value.__class__.__name__ == "PathConfig"

                # If it's a PathConfig and we're not at the top level, just show a reference
                if is_path_config and not _is_top_level:
                    tree.add(
                        Text(
                            f"{field_name}: {value.__class__.__name__}(Singleton)",
                            style="config.value",
                        )
                    )
                    continue

                # Regular singleton handling
                if id(value) in _seen_singletons:
                    tree.add(
                        Text(
                            f"{field_name}: {value.__class__.__name__}(Singleton)",
                            style="config.value",
                        )
                    )
                    continue

                _seen_singletons.add(id(value))
                subtree = tree.add(Text(f"{field_name}:", style=field_style))
                subtree.add(
                    value._build_tree(
                        show_docs=show_docs,
                        _seen_singletons=_seen_singletons,
                        _is_top_level=False,
                        _seen_path_configs=_seen_path_configs,
                    )
                )
                continue

            # Create field node text
            field_text = Text()
            field_text.append(f"{field_name}: ", style=field_style)

            # Handle nested configs
            if isinstance(value, BaseConfig):
                # Special handling for PathConfig
                is_path_config = value.__class__.__name__ == "PathConfig"

                # If it's a PathConfig and we're not at the top level, just show a reference
                if is_path_config and not _is_top_level:
                    tree.add(
                        Text(
                            f"{field_name}: {value.__class__.__name__}(Singleton)",
                            style="config.value",
                        )
                    )
                    continue

                subtree = tree.add(field_text)
                nested_tree = value._build_tree(
                    show_docs=show_docs,
                    _seen_singletons=_seen_singletons,
                    _is_top_level=False,
                    _seen_path_configs=_seen_path_configs,
                )
                subtree.add(nested_tree)
                continue

            # Handle lists/tuples of configs
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], BaseConfig):
                subtree = tree.add(field_text)
                for i, item in enumerate(value):
                    # SingletonConfig handling in lists
                    if isinstance(item, SingletonConfig):
                        # Check if it's a PathConfig
                        is_path_config = item.__class__.__name__ == "PathConfig"

                        # If it's a PathConfig and we're not at the top level, just show a reference
                        if is_path_config and not _is_top_level:
                            subtree.add(
                                Text(
                                    f"[{i}]: {item.__class__.__name__}(Singleton)",
                                    style="config.value",
                                )
                            )
                            continue

                        if id(item) in _seen_singletons:
                            subtree.add(
                                Text(
                                    f"[{i}]: {item.__class__.__name__}(Singleton)",
                                    style="config.value",
                                )
                            )
                            continue
                        _seen_singletons.add(id(item))
                        item_subtree = subtree.add(Text(f"[{i}]:", style="config.field"))
                        item_subtree.add(
                            item._build_tree(
                                show_docs=show_docs,
                                _seen_singletons=_seen_singletons,
                                _is_top_level=False,
                                _seen_path_configs=_seen_path_configs,
                            )
                        )
                        continue

                    # Check if regular item is a PathConfig
                    is_path_config = item.__class__.__name__ == "PathConfig"
                    if is_path_config and not _is_top_level:
                        subtree.add(
                            Text(
                                f"[{i}]: {item.__class__.__name__}(Reference)",
                                style="config.value",
                            )
                        )
                        continue

                    item_tree = item._build_tree(
                        show_docs=show_docs,
                        _seen_singletons=_seen_singletons,
                        _is_top_level=False,
                        _seen_path_configs=_seen_path_configs,
                    )
                    subtree.add(Text(f"[{i}]", style="config.field")).add(item_tree)
                continue

            # Format value
            value_str = self._format_value(value)
            field_text.append(value_str, style="config.value")

            # Add type info
            type_name = self._get_type_name(field.annotation)
            field_text.append(f" ({type_name})", style="config.type")

            # Add field and documentation
            field_node = tree.add(field_text)
            if show_docs and field.description:
                field_node.add(Text(field.description, style="config.doc"))

        return tree

    def _format_value(self, value: Any) -> str:  # pragma: no cover - visualization helper
        """Format a value for display."""
        try:
            if isinstance(value, str):
                return f'"{value}"'
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, Enum):
                return str(value.value if hasattr(value, "value") else value)
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                if not value:
                    return "{}"
                items = [f"{k}: {repr(v)}" for k, v in value.items()]
                return "{" + ", ".join(items) + "}"
            if value is None:
                return "None"
            if isinstance(value, type):
                return value.__name__
            return repr(value)
        except Exception:
            return "<unprintable>"

    def _get_type_name(self, annotation: Any) -> str:  # pragma: no cover - visualization helper
        """Get type name from annotation."""
        try:
            if hasattr(annotation, "__origin__"):
                origin = annotation.__origin__.__name__
                args = []
                for arg in annotation.__args__:
                    if isinstance(arg, ForwardRef):
                        args.append(arg.__forward_arg__)
                    elif hasattr(arg, "__name__"):
                        args.append(arg.__name__)
                    else:
                        args.append(str(arg))
                return f"{origin}[{', '.join(args)}]"
            return str(annotation).replace("typing.", "")
        except Exception:
            return "Any"

    @model_validator(mode="after")
    def _propagate_shared_fields(self) -> "BaseConfig":
        """Propagate shared field values to nested BaseConfig instances."""
        for field_name, field_value in self:
            if field_name in {"propagated_fields", "target"}:
                continue

            if isinstance(field_value, BaseConfig):
                self._propagate_to_child(field_name, field_value)

            elif isinstance(field_value, (list, tuple)):
                for item in field_value:
                    if isinstance(item, BaseConfig):
                        self._propagate_to_child(field_name, item)

        return self

    def _propagate_to_child(self, parent_field: str, child_config: "BaseConfig") -> None:
        """Propagate matching fields from parent to child config"""
        shared_fields = {
            name: value
            for name, value in self
            if name in child_config.__class__.model_fields
            and name != parent_field
            and name not in ("propagated_fields", "target")
        }

        for name, value in shared_fields.items():
            current_value = getattr(child_config, name, None)
            if current_value != value:
                object.__setattr__(child_config, name, value)
                child_config.propagated_fields[name] = value

                Console().log(
                    f"Propagated {name}={value} from {self.__class__.__name__} to {child_config.__class__.__name__}"
                )

    # ------------------------------------------------------------------ TOML utils
    def _write_toml_fields(  # pragma: no cover - serialization helper
        self,
        *,
        container: TOMLDocument | Table,
        include_comments: bool,
        include_type_hints: bool,
    ) -> None:
        for field_name, field in self.__class__.model_fields.items():
            if getattr(field, "exclude", False) or field_name in {"propagated_fields", "target"}:
                continue

            value = getattr(self, field_name)
            if value is None:
                continue
            type_hint: str | None = None
            if include_comments and include_type_hints:
                type_hint = self._get_type_name(field.annotation)

            toml_value = self._to_toml_item(
                value,
                include_comments=include_comments,
                include_type_hints=include_type_hints,
            )

            # Add type hint comment BEFORE the field (or section header for nested configs)
            if type_hint is not None:
                container.add(comment(f"type: {type_hint}"))

            container.add(field_name, toml_value)

    def _to_toml_item(  # pragma: no cover - serialization helper
        self,
        value: Any,
        *,
        include_comments: bool,
        include_type_hints: bool,
    ) -> Any:
        if isinstance(value, BaseConfig):
            nested = table()
            value._write_toml_fields(
                container=nested,
                include_comments=include_comments,
                include_type_hints=include_type_hints,
            )
            return nested

        if isinstance(value, dict):
            tbl = table()
            for key, sub_value in value.items():
                tbl.add(key, self._normalise_scalar(sub_value))
            return tbl

        if isinstance(value, (list, tuple, set)):
            sequence = list(value)
            if all(isinstance(item_value, BaseConfig) for item_value in sequence):
                tables = aot()
                for item_config in sequence:
                    nested = table()
                    item_config._write_toml_fields(
                        container=nested,
                        include_comments=include_comments,
                        include_type_hints=include_type_hints,
                    )
                    tables.append(nested)
                return tables

            arr = array()
            arr.multiline(True)
            for item_value in sequence:
                if isinstance(item_value, BaseConfig):
                    nested = table()
                    item_value._write_toml_fields(
                        container=nested,
                        include_comments=include_comments,
                        include_type_hints=include_type_hints,
                    )
                    arr.append(nested)
                else:
                    arr.append(self._normalise_scalar(item_value))
            return arr

        return self._normalise_scalar(value)

    def _normalise_scalar(self, value: Any) -> Any:
        if isinstance(value, Path):
            return string(str(value))
        if isinstance(value, Enum):
            enum_value = value.value if hasattr(value, "value") else str(value)
            # Preserve numeric types for IntEnum/FloatEnum
            if isinstance(enum_value, (int, float)):
                return enum_value
            return string(str(enum_value))
        return value


class SingletonConfig(BaseConfig):
    """Base class for singleton configurations."""

    _instances: ClassVar[Dict[Type, Any]] = {}
    _lock: ClassVar[Lock] = Lock()

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, validate_default=True
    )

    def __new__(cls, *args: Any, **kwargs: Any):
        with cls._lock:
            if cls not in cls._instances:
                instance = super(BaseConfig, cls).__new__(cls)
                instance.__dict__["_initialized"] = False
                cls._instances[cls] = instance
            return cls._instances[cls]

    def __init__(self, **kwargs):
        if not getattr(self, "_initialized", False):
            super().__init__(**kwargs)
            self.__dict__["_initialized"] = True
        else:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    current = getattr(self, key)
                    if current != value:
                        Console().warn(
                            f"Updating singleton {self.__class__.__name__} field '{key}' from {current} to {value}"
                        )
                    setattr(self, key, value)

    def __copy__(self) -> "SingletonConfig":
        """Return self since this is a singleton."""
        return self

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> "SingletonConfig":
        """Return self since this is a singleton. Implements proper deepcopy protocol."""
        if memo is not None:
            memo[id(self)] = self
        return self
