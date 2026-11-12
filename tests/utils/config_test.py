# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Protocol, runtime_checkable

import pytest
import yaml
from pydantic import Field, ValidationError

from jaqmc.utils.config import (
    ModuleConfig,
    config_to_dict,
    config_to_yaml,
    configurable_dataclass,
    is_configurable_dataclass,
    parse_config,
)
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class Nested:
    count: int = 1
    enabled: bool = False


@configurable_dataclass
class Schema:
    nested: Nested = field(default_factory=Nested)
    values: tuple[int, ...] = ()


@configurable_dataclass
class Leaf:
    rate: float = 0.5


@configurable_dataclass
class Steps:
    steps: int = 1


@configurable_dataclass
class Tuned:
    rate: float = 0.5
    extra: int = 1


@dataclass
class UndecoratedSubclass(Leaf):
    extra: int = 1


@configurable_dataclass
class SwitchedTarget:
    nested: Nested = field(default_factory=lambda: Nested(count=7))
    sub: Annotated[Any, ModuleConfig()] = field(
        default_factory=lambda: Tuned(rate=2.0, extra=9)
    )


@runtime_checkable
class HasRun(Protocol):
    def run(self) -> None: ...


@configurable_dataclass
class Runnable:
    def run(self) -> None:
        pass


def test_parse_config_merges_layers_and_dotlist_without_mutating_inputs():
    first = {"nested": {"count": 2}, "values": ["1", 2]}
    second = {"nested": {"enabled": True}}

    value = parse_config(Schema, [first, second], ["nested.count=4"])

    assert value == Schema(nested=Nested(count=4, enabled=True), values=(1, 2))
    assert first == {"nested": {"count": 2}, "values": ["1", 2]}
    assert second == {"nested": {"enabled": True}}


def test_parse_config_merges_over_customized_factory_defaults():
    @configurable_dataclass
    class CustomizedSchema:
        nested: Nested = field(default_factory=lambda: Nested(count=7))

    value = parse_config(CustomizedSchema, {"nested": {"enabled": True}})

    assert value.nested == Nested(count=7, enabled=True)


def test_parse_config_skips_post_init_while_materializing_defaults():
    validated_counts = []

    @configurable_dataclass
    class ValidatedSchema:
        count: int = 0
        doubled: int = 0

        def __post_init__(self) -> None:
            validated_counts.append(self.count)
            if self.count <= 0:
                raise ValueError("count must be positive")
            self.doubled = 2 * self.count

    value = parse_config(ValidatedSchema, {"count": 3})

    assert value.count == 3
    assert value.doubled == 6
    assert validated_counts == [3]
    with pytest.raises(ValidationError, match="count must be positive"):
        parse_config(ValidatedSchema, {})


def test_configurable_dataclass_guards_inherited_post_init():
    @dataclass
    class ValidatedBase:
        count: int = 0

        def __post_init__(self) -> None:
            if self.count <= 0:
                raise ValueError("count must be positive")

    @configurable_dataclass
    class InheritedSchema(ValidatedBase):
        pass

    assert parse_config(InheritedSchema, {"count": 3}).count == 3
    with pytest.raises(ValidationError, match="count must be positive"):
        parse_config(InheritedSchema, {})


def test_parse_config_uses_pydantic_lax_coercion():
    assert parse_config(Schema, {"nested": {"count": "2"}}).nested.count == 2
    assert parse_config(Schema, {"nested": {"count": 2.0}}).nested.count == 2

    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        parse_config(Schema, {"unknown": 1})
    with pytest.raises(ValidationError, match="fractional part"):
        parse_config(Schema, {"nested": {"count": 2.1}})
    with pytest.raises(ValidationError, match="valid boolean"):
        parse_config(Schema, {"nested": {"enabled": "not_a_bool"}})
    with pytest.raises(ValidationError, match="valid boolean"):
        parse_config(Schema, {"nested": {"enabled": 2}})


def test_parse_config_rejects_non_str_for_str_fields():
    @configurable_dataclass
    class Named:
        name: str = "hi"

    with pytest.raises(ValidationError, match="valid string"):
        parse_config(Named, {"name": 123})
    with pytest.raises(ValidationError, match="valid string"):
        parse_config(Named, {"name": True})


def test_parse_config_reports_non_constructible_defaults():
    @configurable_dataclass
    class RequiredSchema:
        value: int

    with pytest.raises(ValidationError, match="Field required"):
        parse_config(RequiredSchema, {"value": 1})


def test_config_serialization_round_trips_field_alias():
    @configurable_dataclass
    class Renamed:
        count: int = Field(alias="atom_count", default=1)
        other: int = 0

    assert Renamed(atom_count=3).count == 3
    assert replace(Renamed(atom_count=3), other=2) == Renamed(atom_count=3, other=2)
    value = parse_config(Renamed, {"atom_count": "2"})
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        parse_config(Renamed, {"count": 2})

    text = config_to_yaml(value)
    assert yaml.safe_load(text) == {"atom_count": 2, "other": 0}
    assert parse_config(Renamed, yaml.safe_load(text)) == value


def test_module_config_supports_modules_scalars_and_relative_paths(mocker):
    @configurable_dataclass
    class ModuleSchema:
        choice: Annotated[float | Leaf, ModuleConfig()] = field(default_factory=Leaf)

    assert parse_config(ModuleSchema, {"choice": 1.5}).choice == pytest.approx(1.5)
    assert parse_config(ModuleSchema, {"choice": {"rate": 1.25}}).choice == Leaf(
        rate=1.25
    )

    resolve = mocker.patch("jaqmc.utils.config.resolve_object", return_value=Leaf)

    @configurable_dataclass
    class RelativeSchema:
        choice: Annotated[Any, ModuleConfig(package="custom.base")] = field(
            default_factory=Leaf
        )

    assert (
        parse_config(RelativeSchema, {"choice": {"module": "relative:Leaf"}}).choice
        == Leaf()
    )
    resolve.assert_any_call("relative:Leaf", package="custom.base")
    assert {call.kwargs["package"] for call in resolve.call_args_list} == {
        "custom.base"
    }


@pytest.mark.parametrize("raw", [1.5, [0.25, 0.75], {"rate": 1.25}])
def test_module_config_round_trips_direct_values_and_modules(raw):
    @configurable_dataclass
    class DirectSchema:
        choice: Annotated[float | tuple[float, float] | Leaf, ModuleConfig()] = field(
            default_factory=Leaf
        )

    value = parse_config(DirectSchema, {"choice": raw})

    assert parse_config(DirectSchema, yaml.safe_load(config_to_yaml(value))) == value


def test_module_collection_uses_defaults():
    @configurable_dataclass
    class CollectionSchema:
        modules: dict[str, Annotated[Any | None, ModuleConfig()]] = field(
            default_factory=lambda: {"default": Tuned(rate=3, extra=7)}
        )

    value = parse_config(
        CollectionSchema,
        {"modules": {"default": {"rate": 4}, "disabled": None}},
    )

    assert value.modules == {"default": Tuned(rate=4, extra=7), "disabled": None}
    assert (
        parse_config(CollectionSchema, yaml.safe_load(config_to_yaml(value))) == value
    )
    switched = parse_config(
        CollectionSchema,
        [
            {"modules": {"default": {"rate": 4}}},
            {"modules": {"default": {"module": f"{__name__}:Steps"}}},
        ],
    )
    assert switched.modules["default"] == Steps()
    disabled = parse_config(
        CollectionSchema,
        {"modules": {"default": None}},
    )
    assert disabled.modules == {"default": None}


def test_module_collection_resolves_relative_modules_and_validates_defaults():
    package, _, module = __name__.rpartition(".")

    @configurable_dataclass
    class CollectionSchema:
        modules: dict[str, Annotated[Any | None, ModuleConfig(package=package)]] = (
            field(default_factory=lambda: {"leaf": Leaf()})
        )

    value = parse_config(
        CollectionSchema, {"modules": {"extra": {"module": f"{module}:Steps"}}}
    )

    assert value.modules == {"leaf": Leaf(), "extra": Steps()}
    with pytest.raises(
        ValidationError,
        match=r"modules\.new[\s\S]*Missing required config key 'module'",
    ):
        parse_config(CollectionSchema, {"modules": {"new": {"rate": 1.0}}})


def test_same_module_merges_layers_and_dotlist_switch_resets_arguments():
    @configurable_dataclass
    class ModuleSchema:
        choice: Annotated[Any, ModuleConfig()] = field(default_factory=Leaf)

    same = parse_config(
        ModuleSchema,
        [
            {"choice": {"rate": 1.25}},
            {"choice": {"module": f"{__name__}:Leaf"}},
        ],
    )
    switched = parse_config(
        ModuleSchema,
        {"choice": {"rate": 1.25}},
        [f"choice.module={__name__}:Steps"],
    )

    assert same.choice == Leaf(rate=1.25)
    assert switched.choice == Steps()


def test_relative_spelling_of_same_module_keeps_arguments():
    package, _, relative = __name__.rpartition(".")

    @configurable_dataclass
    class ModuleSchema:
        choice: Annotated[Any, ModuleConfig(package=package)] = field(
            default_factory=lambda: Leaf(rate=2.0)
        )

    same = parse_config(
        ModuleSchema, {"choice": {"rate": 1.25}}, [f"choice.module={relative}:Leaf"]
    )
    preset = parse_config(ModuleSchema, {}, [f"choice.module={relative}:Leaf"])
    switched = parse_config(
        ModuleSchema, {"choice": {"rate": 1.25}}, [f"choice.module={relative}:Steps"]
    )

    assert same.choice == Leaf(rate=1.25)
    assert preset.choice == Leaf(rate=2.0)
    assert switched.choice == Steps()


def test_repeated_relative_module_layers_keep_arguments():
    """Layering the same relative module reference twice must not crash.

    The first layer stores its relative spelling in the merged data; the
    second layer still resolves it instead of importing the bare name.
    """
    package, _, relative = __name__.rpartition(".")

    @configurable_dataclass
    class ModuleSchema:
        choice: Annotated[Any, ModuleConfig(package=package)] = field(
            default_factory=Leaf
        )

    reference = f"{relative}:Leaf"
    merged = parse_config(
        ModuleSchema,
        [
            {"choice": {"module": reference, "rate": 1.25}},
            {"choice": {"module": reference}},
        ],
    )

    assert merged.choice == Leaf(rate=1.25)


def _write_plugin(path, rate: float, extra: int) -> str:
    path.write_text(
        "from jaqmc.utils.config import configurable_dataclass\n"
        "\n"
        "@configurable_dataclass\n"
        "class Plugin:\n"
        f"    rate: float = {rate}\n"
        f"    extra: int = {extra}\n",
        encoding="utf-8",
    )
    return f"{path}:Plugin"


def test_file_modules_merge_across_layers_and_survive_serialization(tmp_path):
    plugin = tmp_path / "plugin.py"
    reference = _write_plugin(plugin, rate=0.5, extra=1)

    @configurable_dataclass
    class ModuleSchema:
        choice: Annotated[Any, ModuleConfig()] = field(default_factory=Leaf)

    try:
        merged = parse_config(
            ModuleSchema,
            [
                {"choice": {"module": reference, "rate": 1.25}},
                {"choice": {"module": reference, "extra": 7}},
            ],
        )
        assert (merged.choice.rate, merged.choice.extra) == (1.25, 7)

        # Dropping the loaded module leaves the serialized reference as the only
        # way back to the plugin, as it would be in a later process.
        serialized = yaml.safe_load(config_to_yaml(merged))
        del sys.modules[str(plugin.resolve())]
        restored = parse_config(ModuleSchema, serialized)

        assert (restored.choice.rate, restored.choice.extra) == (1.25, 7)
    finally:
        sys.modules.pop(str(plugin.resolve()), None)


def test_switching_between_file_modules_discards_arguments(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first_reference = _write_plugin(first, rate=0.5, extra=1)
    second_reference = _write_plugin(second, rate=2.5, extra=3)

    @configurable_dataclass
    class ModuleSchema:
        choice: Annotated[Any, ModuleConfig()] = field(default_factory=Leaf)

    try:
        switched = parse_config(
            ModuleSchema,
            [
                {"choice": {"module": first_reference, "rate": 1.25}},
                {"choice": {"module": second_reference}},
            ],
        )

        assert (switched.choice.rate, switched.choice.extra) == (2.5, 3)
    finally:
        for path in (first, second):
            sys.modules.pop(str(path.resolve()), None)


def test_relative_module_resolves_from_field_after_file_module(tmp_path):
    reference = _write_plugin(tmp_path / "plugin.py", rate=0.5, extra=1)
    package, _, relative = __name__.rpartition(".")

    @configurable_dataclass
    class ModuleSchema:
        choice: Annotated[Any, ModuleConfig(package=package)] = field(
            default_factory=Leaf
        )

    try:
        switched = parse_config(
            ModuleSchema,
            {"choice": {"module": reference, "rate": 1.25}},
            [f"choice.module={relative}:Steps"],
        )
        assert switched.choice == Steps()
    finally:
        sys.modules.pop(str((tmp_path / "plugin.py").resolve()), None)


def test_module_layers_reject_unresolvable_and_reset_on_switch():
    @configurable_dataclass
    class ModuleSchema:
        choice: Annotated[Any, ModuleConfig()] = field(
            default_factory=lambda: Leaf(rate=2.0)
        )

    with pytest.raises(ValidationError, match=r"at 'module': .*missing_module"):
        parse_config(
            ModuleSchema,
            {"choice": {"module": "missing_module:Leaf"}},
            [f"choice.module={__name__}:Leaf"],
        )
    switched_back = parse_config(
        ModuleSchema,
        [
            {"choice": {"rate": 1.25}},
            {"choice": {"module": f"{__name__}:Steps", "steps": 3}},
        ],
        [f"choice.module={__name__}:Leaf"],
    )

    assert switched_back.choice == Leaf(rate=2.0)


def test_module_is_a_reserved_field_name():
    with pytest.raises(TypeError, match="'module' is a reserved config key"):

        @configurable_dataclass
        class Reserved:
            module: int = 0


def test_dotlist_reads_exponent_floats():
    assert parse_config(Leaf, {}, ["rate=1e-3"]).rate == pytest.approx(1e-3)
    assert parse_config(Nested, {}, ["count=1e3"]).count == 1000


def test_dataclass_unions_are_untagged_and_merge_per_variant():
    @configurable_dataclass
    class First:
        old: int = 1

    @configurable_dataclass
    class Second:
        new: int
        final: int = 0

    @configurable_dataclass
    class UnionSchema:
        choice: First | Second = field(default_factory=First)

    same = parse_config(
        UnionSchema,
        [
            {"choice": {"new": 2, "final": 3}},
            {"choice": {"new": 4}},
        ],
    )
    switched = parse_config(
        UnionSchema,
        [
            {"choice": {"old": 5}},
            {"choice": {"new": 2}},
        ],
    )

    assert config_to_dict(UnionSchema()) == {"choice": {"old": 1}}
    assert same.choice == Second(new=4, final=3)
    assert switched.choice == Second(new=2)


def test_optional_dataclass_merges_across_layers_without_variant_reset():
    @configurable_dataclass
    class RequiredPair:
        first: int
        second: int
        choice: Leaf | Steps = field(default_factory=Leaf)

    @configurable_dataclass
    class OptionalSchema:
        item: RequiredPair | None = None

    merged = parse_config(
        OptionalSchema,
        [
            {"item": {"first": 1, "choice": {"rate": 1.5}}},
            {"item": {"second": 2, "choice": {"steps": 4}}},
        ],
    )

    assert merged.item == RequiredPair(first=1, second=2, choice=Steps(steps=4))
    assert parse_config(OptionalSchema, [merged.__dict__, {"item": None}]).item is None


def test_module_config_configured_instance_merges_and_resets():
    @configurable_dataclass
    class InstanceSchema:
        choice: Annotated[Any, ModuleConfig()] = field(
            default_factory=lambda: Tuned(rate=2.0, extra=7)
        )

    first = InstanceSchema()
    second = InstanceSchema()
    first.choice.extra = 9

    assert second.choice == Tuned(rate=2.0, extra=7)
    assert parse_config(InstanceSchema, {"choice": {"rate": 3}}).choice == Tuned(
        rate=3.0, extra=7
    )
    assert (
        parse_config(InstanceSchema, {}, [f"choice.module={__name__}:Steps"]).choice
        == Steps()
    )


def test_module_config_rejects_invalid_targets_and_dotlist():
    @configurable_dataclass
    class DirectSchema:
        choice: Annotated[float, ModuleConfig()] = 0.1

    with pytest.raises(ValidationError, match=r"Missing required config key 'module'"):
        parse_config(DirectSchema, {"choice": {"rate": 1.0}})
    with pytest.raises(ValueError, match="expected the form"):
        parse_config(Schema, {}, ["nested.count"])
    with pytest.raises(ValueError, match="is not a mapping"):
        parse_config(Schema, {}, ["nested=1", "nested.count=2"])


def test_module_config_rejects_non_string_module_values():
    @configurable_dataclass
    class ModuleSchema:
        choice: Annotated[Any, ModuleConfig()] = field(default_factory=Leaf)
        modules: dict[str, Annotated[Any | None, ModuleConfig()]] = field(
            default_factory=dict
        )

    with pytest.raises(
        ValidationError, match=r"choice[\s\S]*at 'module': expected a string, got int"
    ):
        parse_config(ModuleSchema, {"choice": {"module": 123}})
    with pytest.raises(
        ValidationError,
        match=r"modules\.one[\s\S]*at 'module': expected a string, got bool",
    ):
        parse_config(ModuleSchema, {"modules": {"one": {"module": True}}})


def test_module_contracts_reject_wrong_modules_without_runtime_building():
    @configurable_dataclass
    class ContractSchema:
        choice: Annotated[HasRun, ModuleConfig()] = field(default_factory=Runnable)
        modules: dict[str, Annotated[HasRun | None, ModuleConfig()]] = field(
            default_factory=dict
        )

    parsed = parse_config(
        ContractSchema, {"modules": {"one": {"module": f"{__name__}:Runnable"}}}
    )
    assert isinstance(parsed.choice, Runnable)
    assert isinstance(parsed.modules["one"], Runnable)

    with pytest.raises(ValidationError, match="instance of HasRun"):
        parse_config(
            ContractSchema,
            {"choice": {"module": f"{__name__}:Leaf"}},
        )
    with pytest.raises(ValidationError, match="instance of HasRun"):
        parse_config(
            ContractSchema,
            {"modules": {"one": {"module": f"{__name__}:Leaf"}}},
        )
    with pytest.raises(ValidationError, match="instance of HasRun"):
        ContractSchema(choice=Leaf())  # type: ignore[arg-type]


def test_switching_module_merges_over_new_target_nested_defaults():
    @configurable_dataclass
    class ParentSchema:
        choice: Annotated[Any, ModuleConfig()] = field(default_factory=Leaf)

    parsed = parse_config(
        ParentSchema,
        {
            "choice": {
                "module": f"{__name__}:SwitchedTarget",
                "nested": {"enabled": True},
                "sub": {"rate": 0.25},
            }
        },
    )

    assert parsed.choice == SwitchedTarget(
        nested=Nested(count=7, enabled=True),
        sub=Tuned(rate=0.25, extra=9),
    )


def test_wavefunction_fields_validate_types():
    from jaqmc.app.molecule.workflow import MoleculeTrainWorkflow

    with pytest.raises(ValidationError, match="fractional part"):
        MoleculeTrainWorkflow.from_config({"wf": {"ndets": 2.1}})
    with pytest.raises(ValidationError, match="valid integer"):
        MoleculeTrainWorkflow.from_config({"wf": {"ndets": "not_an_int"}})


def test_hidden_fields_allow_arbitrary_annotations():
    class CustomDep:
        pass

    @configurable_dataclass
    class HiddenAnnotationSchema:
        choice: Annotated[CustomDep | Leaf, ModuleConfig()] = field(
            default_factory=Leaf
        )
        collection: dict[str, Annotated[CustomDep | Leaf | None, ModuleConfig()]] = (
            field(default_factory=lambda: {"leaf": Leaf()})
        )
        dep: CustomDep | None = runtime_dep(default=None)

    dep = CustomDep()
    assert HiddenAnnotationSchema(dep=dep).dep is dep
    parsed = parse_config(HiddenAnnotationSchema, {"choice": {"rate": 2.0}})
    assert parsed.choice == Leaf(rate=2.0)
    assert parsed.collection == {"leaf": Leaf()}
    assert parsed.dep is None
    assert "dep" not in config_to_dict(parsed)
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        parse_config(HiddenAnnotationSchema, {"dep": dep})


def test_undecorated_dataclass_subclass_is_rejected():
    @configurable_dataclass
    class ModuleSchema:
        choice: Annotated[Any, ModuleConfig()] = field(default_factory=Leaf)

    assert not is_configurable_dataclass(UndecoratedSubclass)
    assert not is_configurable_dataclass(UndecoratedSubclass())
    with pytest.raises(ValueError, match="not a configurable dataclass"):
        parse_config(UndecoratedSubclass, {})
    with pytest.raises(ValueError, match="not a configurable dataclass"):
        config_to_dict(UndecoratedSubclass())
    with pytest.raises(ValidationError, match="not a configurable dataclass"):
        parse_config(
            ModuleSchema, {"choice": {"module": f"{__name__}:UndecoratedSubclass"}}
        )
