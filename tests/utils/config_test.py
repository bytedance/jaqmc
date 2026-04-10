# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pytest
import yaml

from jaqmc.utils.config import ConfigManager, configurable_dataclass, module_config
from jaqmc.workflow.base import Workflow


@dataclass
class SimpleConfig:
    x: int = 1
    y: str = "default"


@dataclass
class OtherConfig:
    z: int = 7


@configurable_dataclass
class Axis:
    direction: tuple[float, ...] = (0.0, 0.0, 1.0)
    bins: int = 50
    range: tuple[float, float] = (0.0, 1.0)


@configurable_dataclass
class DictOfOptionalDataclass:
    axes: dict[str, Axis | None] = field(default_factory=dict)


def simple_func(a: int = 10, b: int = 20):
    return a, b


def another_simple_func(a: int = 10, b: int = 20):
    return b, a


class PresetWorkflow(Workflow):
    @classmethod
    def default_preset(cls) -> dict[str, Any]:
        return {
            "workflow": {"disable_jit": True},
            "section": {"x": 5},
        }

    def run(self) -> None:
        raise NotImplementedError()


def test_get_primitive():
    cfg = ConfigManager({"a": 1, "b": "2"}, dotlist=["c=3.0"])

    # First access
    assert cfg.get("a", default=0) == 1
    assert cfg.get("b", default="") == "2"
    assert cfg.get("c", default=0.0) == pytest.approx(3.0)
    assert cfg.get("d", default=100) == 100

    # Idempotency checks
    assert cfg.get("a", default=0) == 1
    assert cfg.get("b", default="") == "2"
    assert cfg.get("c", default=0.0) == pytest.approx(3.0)

    cfg.finalize()


def test_get_primitive_multi_config():
    cfg = ConfigManager(({"a": 3, "b": "2"}, {"a": 1}), dotlist=["c=3.0"])

    assert cfg.get("a", default=0) == 1
    assert cfg.get("b", default="") == "2"
    assert cfg.get("c", default=0.0) == pytest.approx(3.0)

    cfg.finalize()


def test_get_primitive_preset():
    cfg = ConfigManager({"a": 3})
    cfg.use_preset({"a": 5, "b": 100})

    assert cfg.get("a", default=0) == 3
    assert cfg.get("b", default=0) == 100

    cfg.finalize()


def test_workflow_default_preset_applied():
    cfg = ConfigManager({})
    wf = PresetWorkflow(cfg)

    assert wf.config.disable_jit is True
    assert cfg.get("section", default=SimpleConfig()).x == 5

    cfg.finalize()


def test_workflow_default_preset_is_low_priority():
    cfg = ConfigManager(
        {"workflow": {"disable_jit": False}, "section": {"x": 9}},
        dotlist=["section.x=11"],
    )
    wf = PresetWorkflow(cfg)

    assert wf.config.disable_jit is False
    assert cfg.get("section", default=SimpleConfig()).x == 11

    cfg.finalize()


def test_get_dataclass():
    data = {"section": {"x": 10}}
    cfg = ConfigManager(data)

    # First access
    result1 = cfg.get("section", default=SimpleConfig(x=5))
    assert result1.x == 10
    assert result1.y == "default"

    # Idempotency check
    result2 = cfg.get("section", default=SimpleConfig(x=5))
    assert result2 == result1

    cfg.finalize()


def test_get_callable():
    data = {"func": {"a": 5}}
    cfg = ConfigManager(data)

    # First access
    partial_func = cfg.get("func", default=simple_func)
    assert partial_func() == (5, 20)

    # Idempotency check
    partial_func2 = cfg.get("func", default=simple_func)
    assert partial_func2() == (5, 20)

    cfg.finalize()


def test_get_module(mocker):
    mocker.patch("jaqmc.utils.config.resolve_object", return_value=simple_func)

    data = {"my_module": {"module": "some.path", "a": 30}}
    cfg = ConfigManager(data)

    func = cfg.get_module("my_module", default_module="unused")
    assert func() == (30, 20)
    assert func(a=10) == (10, 20)

    # Idempotency check
    func_idempotent = cfg.get_module("my_module", default_module="unused")
    assert func_idempotent() == (30, 20)
    assert func_idempotent(a=10) == (10, 20)

    cfg.finalize()


def test_get_module_non_str_default(mocker):
    data = {"my_module": {"a": 30}}
    cfg = ConfigManager(data)

    func = cfg.get_module("my_module", default_module=simple_func)
    assert func() == (30, 20)
    assert func(a=10) == (10, 20)

    cfg.finalize()


def test_get_module_non_str_default_get(mocker):
    mocker.patch("jaqmc.utils.config.resolve_object", return_value=simple_func)
    data = {"my_module": {"module": "some.path", "a": 30}}
    cfg = ConfigManager(data)

    func = cfg.get_module("my_module", default_module=another_simple_func)
    assert func() == (30, 20)
    assert func(a=10) == (10, 20)

    cfg.finalize()


def test_get_module_dataclass(mocker):
    mocker.patch("jaqmc.utils.config.resolve_object", return_value=SimpleConfig)

    data = {"my_module": {"module": "some.path", "x": 30}}
    cfg = ConfigManager(data)

    result1 = cfg.get_module("my_module", default_module="unused")
    assert isinstance(result1, SimpleConfig)
    assert result1.x == 30

    # Idempotency check
    result2 = cfg.get_module("my_module", default_module="unused")
    assert isinstance(result2, SimpleConfig)
    assert result2.x == 30

    cfg.finalize()


def test_get_collection(mocker):
    mocker.patch("jaqmc.utils.config.resolve_object", return_value=simple_func)

    data = {"collection": {"item1": {"a": 3}, "item2": None}}
    cfg = ConfigManager(data)

    results = cfg.get_collection(
        "collection",
        defaults={
            "item1": {"module": "path1", "a": 1},
            "item2": {"module": "path2", "a": 2},
        },
    )

    assert "item1" in results
    assert results["item1"] == (3, 20)
    assert "item2" not in results

    # Idempotency check
    results = cfg.get_collection(
        "collection",
        defaults={
            "item1": {"module": "path1", "a": 1},
            "item2": {"module": "path2", "a": 2},
        },
    )
    assert results["item1"] == (3, 20)
    assert "item2" not in results

    cfg.finalize()


def test_get_collection_override_none(mocker):
    mocker.patch("jaqmc.utils.config.resolve_object", return_value=simple_func)

    data = [
        {"collection": {"item1": {"a": 3}, "item2": None}},
        {"collection": {"item2": {"a": 4}}},
    ]
    cfg = ConfigManager(data)

    results = cfg.get_collection(
        "collection",
        defaults={
            "item1": {"module": "path1", "a": 1},
            "item2": {"module": "path2", "a": 2},
        },
    )

    assert "item1" in results
    assert results["item1"] == (3, 20)
    assert "item2" in results
    assert results["item2"] == (4, 20)

    cfg.finalize()


def test_get_collection_dotlist_override(mocker):
    mocker.patch("jaqmc.utils.config.resolve_object", return_value=simple_func)
    cfg = ConfigManager({}, dotlist=["collection.item1.a=42", "collection.item1.b=99"])
    results = cfg.get_collection(
        "collection",
        defaults={
            "item1": {"module": "path1"},
        },
    )
    assert results["item1"] == (42, 99)  # Both a and b overridden via dotlist
    cfg.finalize()


def test_get_collection_dataclass_defaults(mocker):
    mocker.patch("jaqmc.utils.config.resolve_object", return_value=SimpleConfig)

    cfg = ConfigManager({"collection": {"item1": {"y": "user"}}})
    results = cfg.get_collection(
        "collection",
        defaults={
            "item1": {"module": "path1", "x": 42},
        },
    )

    assert isinstance(results["item1"], SimpleConfig)
    assert results["item1"].x == 42
    assert results["item1"].y == "user"
    assert cfg.resolved_config["collection"]["item1"] == {
        "module": "path1",
        "x": 42,
        "y": "user",
    }

    cfg.finalize()


def test_get_collection_dataclass_module_override_ignores_old_defaults(mocker):
    mocker.patch(
        "jaqmc.utils.config.resolve_object",
        side_effect=lambda path: {"path1": SimpleConfig, "path2": OtherConfig}[path],
    )

    cfg = ConfigManager({"collection": {"item1": {"module": "path2"}}})
    results = cfg.get_collection(
        "collection",
        defaults={
            "item1": {"module": "path1", "x": 42},
        },
    )

    assert isinstance(results["item1"], OtherConfig)
    assert results["item1"].z == 7
    assert cfg.resolved_config["collection"]["item1"] == {
        "module": "path2",
        "z": 7,
    }

    cfg.finalize()


def test_finalize_unused():
    cfg = ConfigManager({"used": 1, "unused": 2})
    cfg.get("used", 0)

    # Should exit if unused keys exist and raise_on_unused is True (default)
    with pytest.raises(SystemExit):
        cfg.finalize()

    # Should not raise if we ignore it
    cfg.finalize(raise_on_unused=False)


def test_to_yaml_with_comments():
    @dataclass
    class MyConfig:
        """My Docstring."""

        x: int = 1

    cfg = ConfigManager({})
    cfg.get("section", MyConfig)
    cfg.get("val", 10)

    yaml_str = cfg.to_yaml()
    lines = yaml_str.splitlines()

    # Dataclass keys get a "# Defined in" source comment
    section_idx = next(i for i, line in enumerate(lines) if "section:" in line)
    assert "# Defined in" in lines[section_idx - 1]
    assert "config_test.py" in lines[section_idx - 1]

    # Primitive keys do NOT get a source comment
    val_idx = next(i for i, line in enumerate(lines) if "val:" in line)
    assert "# Defined in" not in lines[val_idx - 1]

    # Verbose output includes docstrings
    yaml_verbose = cfg.to_yaml(verbose=True)
    assert "# My Docstring" in yaml_verbose


def test_to_yaml_nested_dataclass():
    @dataclass
    class Inner:
        x: int = 1

    cfg = ConfigManager({})
    cfg.get("a.b", Inner)

    yaml_str = cfg.to_yaml()
    lines = yaml_str.splitlines()

    # The nested dataclass key should have a source comment
    b_idx = next(i for i, line in enumerate(lines) if "b:" in line)
    assert "# Defined in" in lines[b_idx - 1]
    assert "config_test.py" in lines[b_idx - 1]


def test_to_yaml_nested_primitive():
    cfg = ConfigManager({})
    cfg.get("a.b", 100)

    yaml_str = cfg.to_yaml()
    assert "a:" in yaml_str
    assert "b: 100" in yaml_str
    # Primitives do not get source comments
    assert "# Defined in" not in yaml_str


def test_ignore_module_key_source(mocker):
    mocker.patch("jaqmc.utils.config.resolve_object", return_value=simple_func)

    cfg = ConfigManager({})
    # This triggers get_primitive("my_component.module")
    cfg.get_module("my_component", default_module="some.path")

    yaml_str = cfg.to_yaml()

    # We expect my_component to be present
    assert "my_component:" in yaml_str
    assert "module: some.path" in yaml_str

    lines = yaml_str.splitlines()
    for i, line in enumerate(lines):
        if "module: some.path" in line and i > 0:
            # Ensure the previous line is NOT a definition comment
            prev_line = lines[i - 1].strip()
            assert not prev_line.startswith("# Defined in"), (
                f"Found unexpected source comment for module key: {prev_line}"
            )


@dataclass
class InnerClass:
    x: int = 100


@dataclass
class InnerClass2:
    y: int = 100


@dataclass
class MidClass:
    inner: Any = module_config(InnerClass)
    one: Any = module_config(InnerClass, x=1)
    another: Any = module_config(InnerClass, x=2)


@dataclass
class OuterClass:
    mid: Any = module_config(MidClass)


def _module_ref(obj: type | Any) -> str:
    return f"{__name__}:{obj.__name__}"


@pytest.fixture
def recursive_module_mock(mocker):
    def mock_resolve(name, package=None):
        if name == _module_ref(InnerClass):
            return InnerClass
        if name == _module_ref(InnerClass2):
            return InnerClass2
        if name == _module_ref(MidClass):
            return MidClass
        if name == _module_ref(OuterClass):
            return OuterClass
        raise ImportError(f"Cannot resolve {name}")

    mocker.patch("jaqmc.utils.config.resolve_object", side_effect=mock_resolve)


def test_recursive_module_instantiation(recursive_module_mock):
    cfg = ConfigManager(
        {
            "outer": {
                "module": _module_ref(OuterClass),
                "mid": {
                    "module": _module_ref(MidClass),
                    "inner": {
                        "module": _module_ref(InnerClass),
                        "x": 999,
                    },
                },
            }
        }
    )

    outer_instance = cfg.get_module("outer", default_module=_module_ref(OuterClass))

    assert isinstance(outer_instance, OuterClass)
    assert isinstance(outer_instance.mid, MidClass)
    assert isinstance(outer_instance.mid.inner, InnerClass)
    assert outer_instance.mid.inner.x == 999
    assert outer_instance.mid.one.x == 1
    assert outer_instance.mid.another.x == 2

    cfg.finalize()


def test_nesed_module_missing_field(recursive_module_mock):
    cfg = ConfigManager(
        {
            "mid": {
                "module": _module_ref(MidClass),
                "inner": {
                    "module": _module_ref(InnerClass),
                    "x": 999,
                },
                "one": {
                    "module": _module_ref(InnerClass2),
                    "y": 999,
                },
            }
        }
    )

    mid_instance = cfg.get_module("mid", default_module=_module_ref(MidClass))

    assert isinstance(mid_instance, MidClass)
    assert isinstance(mid_instance.inner, InnerClass)
    assert isinstance(mid_instance.one, InnerClass2)
    assert mid_instance.inner.x == 999
    assert mid_instance.one.y == 999

    cfg.finalize()


# --- YAML round-trip tests (resume-from-yaml scenario) ---


@dataclass
class TupleConfig:
    spins: tuple[int, int] = (1, 0)
    basis: str = "sto-3g"


class _TestUnit(StrEnum):
    bohr = "bohr"
    angstrom = "angstrom"


def _func_with_enum(unit: str = _TestUnit.bohr, scale: float = 1.0):
    return unit, scale


def _yaml_roundtrip(
    cfg: ConfigManager, dotlist: list[str] | None = None
) -> ConfigManager:
    """Dump resolved config to YAML, parse back, return new ConfigManager.

    Returns:
        A new ConfigManager initialized from the YAML output.
    """
    yaml_str = cfg.to_yaml()
    parsed = yaml.safe_load(yaml_str)
    return ConfigManager(parsed, dotlist=dotlist)


class TestYamlRoundTrip:
    """Resolved YAML can be parsed back to produce identical configs.

    This simulates the resume scenario: first run dumps config.yaml,
    second run loads it via ``--yml config.yaml`` and optionally
    overrides values with dotlist (e.g. ``train.run.iterations=200``).
    """

    def test_primitives(self):
        cfg1 = ConfigManager({"a": 1, "b": "hello", "c": 1.23})
        a1 = cfg1.get("a", default=0)
        b1 = cfg1.get("b", default="")
        c1 = cfg1.get("c", default=0.0)

        cfg2 = _yaml_roundtrip(cfg1)
        assert cfg2.get("a", default=0) == a1
        assert cfg2.get("b", default="") == b1
        assert cfg2.get("c", default=0.0) == c1
        cfg2.finalize()

    def test_dataclass(self):
        cfg1 = ConfigManager({"section": {"x": 42}})
        r1 = cfg1.get("section", default=SimpleConfig())
        assert r1.x == 42
        assert r1.y == "default"

        cfg2 = _yaml_roundtrip(cfg1)
        r2 = cfg2.get("section", default=SimpleConfig())
        assert r2.x == r1.x
        assert r2.y == r1.y
        cfg2.finalize()

    def test_dataclass_with_tuple(self):
        """Tuples survive YAML round-trip (YAML lists coerced back to tuples)."""
        cfg1 = ConfigManager({"sys": {"spins": [3, 2]}})
        r1 = cfg1.get("sys", default=TupleConfig())
        assert r1.spins == (3, 2)

        cfg2 = _yaml_roundtrip(cfg1)
        r2 = cfg2.get("sys", default=TupleConfig())
        assert r2.spins == (3, 2)
        assert r2.basis == "sto-3g"
        cfg2.finalize()

    def test_callable_module(self, mocker):
        mocker.patch("jaqmc.utils.config.resolve_object", return_value=simple_func)

        cfg1 = ConfigManager({"fn": {"module": "test:simple_func", "a": 42}})
        f1 = cfg1.get_module("fn", default_module="test:simple_func")
        assert f1() == (42, 20)

        cfg2 = _yaml_roundtrip(cfg1)
        f2 = cfg2.get_module("fn", default_module="test:simple_func")
        assert f2() == (42, 20)
        cfg2.finalize()

    def test_callable_with_enum_default(self, mocker):
        """StrEnum defaults survive YAML round-trip (serialized as plain strings)."""
        mocker.patch("jaqmc.utils.config.resolve_object", return_value=_func_with_enum)

        cfg1 = ConfigManager({})
        f1 = cfg1.get_module("fn", default_module="test:func_with_enum")
        assert f1() == ("bohr", 1.0)

        cfg2 = _yaml_roundtrip(cfg1)
        f2 = cfg2.get_module("fn", default_module="test:func_with_enum")
        assert f2() == ("bohr", 1.0)
        cfg2.finalize()

    def test_nested_module_config(self, recursive_module_mock):
        cfg1 = ConfigManager(
            {
                "outer": {
                    "module": _module_ref(OuterClass),
                    "mid": {
                        "inner": {"x": 999},
                    },
                }
            }
        )
        r1 = cfg1.get_module("outer", default_module=_module_ref(OuterClass))
        assert r1.mid.inner.x == 999

        cfg2 = _yaml_roundtrip(cfg1)
        r2 = cfg2.get_module("outer", default_module=_module_ref(OuterClass))
        assert r2.mid.inner.x == 999
        assert r2.mid.one.x == 1
        assert r2.mid.another.x == 2
        cfg2.finalize()

    def test_collection(self, mocker):
        mocker.patch("jaqmc.utils.config.resolve_object", return_value=simple_func)

        defaults = {"item1": {"module": "path1", "a": 1}}
        cfg1 = ConfigManager({"coll": {"item1": {"a": 3}}})
        r1 = cfg1.get_collection("coll", defaults=defaults)
        assert r1["item1"] == (3, 20)

        cfg2 = _yaml_roundtrip(cfg1)
        r2 = cfg2.get_collection("coll", defaults=defaults)
        assert r2["item1"] == (3, 20)
        cfg2.finalize()

    def test_collection_dataclass_defaults(self, mocker):
        mocker.patch("jaqmc.utils.config.resolve_object", return_value=SimpleConfig)

        defaults = {"item1": {"module": "path1", "x": 42}}
        cfg1 = ConfigManager({"coll": {"item1": {"y": "user"}}})
        r1 = cfg1.get_collection("coll", defaults=defaults)
        assert r1["item1"].x == 42
        assert r1["item1"].y == "user"

        cfg2 = _yaml_roundtrip(cfg1)
        r2 = cfg2.get_collection("coll", defaults=defaults)
        assert r2["item1"].x == 42
        assert r2["item1"].y == "user"
        cfg2.finalize()

    def test_dotlist_override_on_resume(self):
        """Dotlist overrides take effect when resuming from YAML."""
        cfg1 = ConfigManager({"section": {"x": 42, "y": "hello"}})
        cfg1.get("section", default=SimpleConfig())

        cfg2 = _yaml_roundtrip(cfg1, dotlist=["section.x=100"])
        r2 = cfg2.get("section", default=SimpleConfig())
        assert r2.x == 100  # overridden by dotlist
        assert r2.y == "hello"  # preserved from YAML
        cfg2.finalize()

    def test_nested_module_config_dotlist_override(self, recursive_module_mock):
        """Dotlist can override nested module_config fields on resume."""
        cfg1 = ConfigManager(
            {
                "outer": {
                    "module": _module_ref(OuterClass),
                    "mid": {"inner": {"x": 999}},
                }
            }
        )
        cfg1.get_module("outer", default_module=_module_ref(OuterClass))

        cfg2 = _yaml_roundtrip(cfg1, dotlist=["outer.mid.inner.x=777"])
        r2 = cfg2.get_module("outer", default_module=_module_ref(OuterClass))
        assert r2.mid.inner.x == 777  # overridden
        assert r2.mid.one.x == 1  # preserved
        cfg2.finalize()


class TestDictOfOptionalDataclass:
    """cfg.get for dataclasses with dict[str, Dataclass | None] fields."""

    def test_roundtrip_no_user_config(self):
        """Instance default survives cfg.get when no user config exists."""
        cfg = ConfigManager({})
        default = DictOfOptionalDataclass(
            axes={"z": Axis(direction=(0, 0, 1), bins=100, range=(-8, 8))}
        )
        result = cfg.get("density", default)
        assert result.axes["z"] is not None
        assert result.axes["z"].bins == 100
        assert result.axes["z"].range == (-8, 8)

    def test_partial_override(self):
        """User config partially overrides one field of a dict entry."""
        cfg = ConfigManager({"density": {"axes": {"z": {"range": [-6, 6]}}}})
        default = DictOfOptionalDataclass(
            axes={"z": Axis(direction=(0, 0, 1), bins=100, range=(-8, 8))}
        )
        result = cfg.get("density", default)
        assert result.axes["z"] is not None
        assert result.axes["z"].range == (-6, 6)
        assert result.axes["z"].bins == 100  # preserved from default
        assert result.axes["z"].direction == (0, 0, 1)  # preserved

    def test_none_disables_entry(self):
        """Setting a dict entry to None should disable it."""
        cfg = ConfigManager({"density": {"axes": {"z": None}}})
        default = DictOfOptionalDataclass(
            axes={"z": Axis(direction=(0, 0, 1), bins=100, range=(-8, 8))}
        )
        result = cfg.get("density", default)
        assert result.axes["z"] is None

    def test_add_new_entry(self):
        """User config adds a new dict entry not in the default."""
        x_axis = {"direction": [1, 0, 0], "bins": 50, "range": [-5, 5]}
        cfg = ConfigManager({"density": {"axes": {"x": x_axis}}})
        default = DictOfOptionalDataclass(
            axes={"z": Axis(direction=(0, 0, 1), bins=100, range=(-8, 8))}
        )
        result = cfg.get("density", default)
        assert result.axes["z"] is not None
        assert result.axes["x"] is not None
        assert result.axes["x"].direction == (1, 0, 0)
