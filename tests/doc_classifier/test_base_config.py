from __future__ import annotations

import copy
from enum import Enum
from pathlib import Path
from typing import Any, ForwardRef, List

import pytest
from pydantic import Field
from tomlkit import string
from tomlkit.items import Table

from traenslenzor.doc_classifier.configs.path_config import PathConfig
from traenslenzor.doc_classifier.utils.base_config import BaseConfig, SingletonConfig
from traenslenzor.doc_classifier.utils.console import Console


class DummyService:
    def __init__(self, config: "DummyConfig"):
        self.config = config
        self.message = config.message


class DummyConfig(BaseConfig[DummyService]):
    """Minimal config used to test BaseConfig.setup_target."""

    target: type[DummyService] = Field(default=DummyService, exclude=True)
    message: str = "hello"


class ChildConfig(BaseConfig[None]):
    shared: str | None = None
    identifier: str


class ParentConfig(BaseConfig[None]):
    shared: str = "root"
    child: ChildConfig = Field(default_factory=lambda: ChildConfig(identifier="primary"))
    children: list[ChildConfig] = Field(default_factory=list)


class NoTargetConfig(BaseConfig[None]):
    """Config without an explicit target used to exercise NoTarget."""

    value: int = 1


class DemoSingleton(SingletonConfig):
    """Singleton used to verify update flows."""

    name: str = "demo"


class ComplexConfig(BaseConfig[None]):
    """Comprehensive configuration exercising BaseConfig utilities."""

    class Mode(Enum):
        ALPHA = "alpha"

    title: str = "root"
    count: int = 5
    active: bool = True
    path_value: Path = Path("artifacts/model.ckpt")
    enum_value: Mode = Mode.ALPHA
    mapping: dict[str, int] = Field(default_factory=lambda: {"foo": 1})
    empty_mapping: dict[str, Any] = Field(default_factory=dict)
    maybe_none: str | None = None
    child: ChildConfig = Field(default_factory=lambda: ChildConfig(identifier="nested"))
    child_list: list[ChildConfig] = Field(
        default_factory=lambda: [
            ChildConfig(identifier="branch-a"),
            ChildConfig(identifier="branch-b"),
        ]
    )
    singleton: DemoSingleton = Field(default_factory=DemoSingleton, exclude=True)
    other_singleton: DemoSingleton = Field(default_factory=DemoSingleton, exclude=True)
    path_config: PathConfig = Field(default_factory=PathConfig, exclude=True)


def test_setup_target_instantiates_target() -> None:
    config = DummyConfig(message="hi")
    service = config.setup_target()

    assert isinstance(service, DummyService)
    assert service.message == "hi"
    assert service.config is config


def test_shared_fields_propagate_to_nested_configs() -> None:
    parent = ParentConfig(
        shared="propagated",
        child=ChildConfig(shared="override-me", identifier="single"),
        children=[
            ChildConfig(identifier="list-a"),
            ChildConfig(identifier="list-b", shared="should-change"),
        ],
    )

    assert parent.child.shared == "propagated"
    assert parent.child.propagated_fields["shared"] == "propagated"

    for child in parent.children:
        assert child.shared == "propagated"
        assert child.propagated_fields["shared"] == "propagated"


def test_toml_roundtrip(tmp_path) -> None:
    parent = ParentConfig(
        shared="toml-shared",
        children=[
            ChildConfig(identifier="first"),
            ChildConfig(identifier="second"),
        ],
    )

    toml_text = parent.to_toml()
    assert "# type: str" in toml_text
    path = tmp_path / "config.toml"
    parent.to_toml(path=path)

    loaded = ParentConfig.from_toml(path)
    assert loaded.shared == "toml-shared"
    assert len(loaded.children) == 2
    assert loaded.children[0].identifier == "first"


def test_puml_roundtrip(tmp_path) -> None:
    parent = ParentConfig(
        shared="diagram-shared",
        children=[
            ChildConfig(identifier="first"),
            ChildConfig(identifier="second"),
        ],
    )

    puml = parent.to_puml()
    assert "@startuml" in puml
    path = tmp_path / "config.puml"
    parent.to_puml(path=path)

    loaded = ParentConfig.from_puml(path)
    assert loaded.shared == "diagram-shared"
    assert len(loaded.children) == 2


def test_notarget_setup_and_error_branch(monkeypatch) -> None:
    config = NoTargetConfig()
    assert config.setup_target() is None

    object.__setattr__(config, "target", "not-callable")  # bypass validation
    with pytest.raises(ValueError):
        config.setup_target()


def test_to_toml_includes_doc_comments(tmp_path) -> None:
    class DocumentedConfig(BaseConfig[None]):
        """Important config for docs."""

        message: str = "hello"

    config = DocumentedConfig()
    rendered = config.to_toml()
    assert "Important config for docs." in rendered

    path = tmp_path / "documented.toml"
    config.to_toml(path=path)

    from_str = DocumentedConfig.from_toml(str(path))
    from_bytes = DocumentedConfig.from_toml(path.read_bytes())
    assert from_str.message == "hello"
    assert from_bytes.message == "hello"


def test_puml_variants_and_missing_payload(tmp_path) -> None:
    config = ComplexConfig()
    text = config.to_puml(include_values=False)
    assert "title:" not in text  # No attribute values emitted

    path = tmp_path / "graph.puml"
    config.to_puml(path=path)
    reloaded = ComplexConfig.from_puml(path)
    assert reloaded.title == "root"

    with pytest.raises(ValueError):
        ComplexConfig.from_puml("@startuml\n@enduml")


def test_build_tree_and_inspect(monkeypatch) -> None:
    config = ComplexConfig()
    tree = config._build_tree(show_docs=True)
    assert tree.label.plain == "ComplexConfig"

    captured: list[Any] = []
    monkeypatch.setattr(Console, "print", lambda self, value, **_: captured.append(value))
    config.inspect(show_docs=True)
    assert captured  # ensure Console.print invoked


def test_format_value_and_type_name_handling() -> None:
    config = ComplexConfig()
    nodes, edges = config._build_puml_graph(include_values=True)
    assert nodes
    assert edges
    # Exercise _sanitize_puml_name explicitly
    assert config._sanitize_puml_name("ComplexConfig.child[0]") == "ComplexConfig_child_0_"
    assert config._format_value(int) == "int"

    type_name = config._get_type_name(List[ForwardRef("Demo")])
    assert "list" in type_name.lower()


def test_shared_field_propagation_logs(monkeypatch) -> None:
    class SharedParent(BaseConfig[None]):
        shared: str = "parent"
        child: ChildConfig = Field(default_factory=lambda: ChildConfig(identifier="inner"))

    logged: list[str] = []
    monkeypatch.setattr(Console, "log", lambda self, msg: logged.append(msg))
    parent = SharedParent()
    parent._propagate_shared_fields()
    assert "Propagated shared=parent" in logged[0]
    assert parent.child.shared == "parent"


def test_toml_serialisation_variants() -> None:
    config = ComplexConfig()
    doc = config.to_toml()
    assert "mapping" in doc

    # Direct access to private helpers for edge cases
    tbl = config._to_toml_item({"k": 1}, include_comments=False, include_type_hints=False)
    assert isinstance(tbl, Table)
    arr = config._to_toml_item(
        [ChildConfig(identifier="x")], include_comments=False, include_type_hints=False
    )
    assert len(arr) == 1
    mixed = config._to_toml_item(
        [ChildConfig(identifier="y"), 5],
        include_comments=False,
        include_type_hints=False,
    )
    assert len(mixed) == 2

    assert config._normalise_scalar(Path("foo")) == string("foo")
    assert config._normalise_scalar(ComplexConfig.Mode.ALPHA) == string("alpha")


def test_singleton_update_and_copy(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(Console, "warn", lambda self, msg: warnings.append(msg))

    singleton = DemoSingleton()
    DemoSingleton.__init__(singleton, name="override")
    assert warnings

    assert copy.copy(singleton) is singleton
    assert copy.deepcopy(singleton) is singleton
