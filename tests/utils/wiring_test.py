# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from jaqmc.utils.wiring import UNSET, check_wired, runtime_dep, wire


@dataclass
class _Component:
    config_field: str = "default"
    required_dep: object = runtime_dep()
    optional_dep: str = runtime_dep(default="fallback")


@dataclass
class _NoDeps:
    value: int = 0


# --- wire() ---


class TestWire:
    def test_wires_required_dep(self):
        comp = _Component()
        sentinel = object()
        wire(comp, required_dep=sentinel)
        assert comp.required_dep is sentinel

    def test_missing_required_dep_raises(self):
        comp = _Component()
        with pytest.raises(ValueError, match="required_dep"):
            wire(comp, unrelated_key="x")

    def test_optional_dep_keeps_default(self):
        comp = _Component()
        wire(comp, required_dep="x")
        assert comp.optional_dep == "fallback"

    def test_optional_dep_overridden(self):
        comp = _Component()
        wire(comp, required_dep="x", optional_dep="custom")
        assert comp.optional_dep == "custom"

    def test_rejects_non_dataclass(self):
        with pytest.raises(TypeError, match=r"wire.*expects a dataclass"):
            wire("not a dataclass")


# --- check_wired() ---


class TestCheckWired:
    def test_passes_when_all_deps_set(self):
        comp = _Component(required_dep="wired")
        check_wired(comp)  # should not raise

    def test_catches_unwired_required_dep(self):
        comp = _Component()
        with pytest.raises(ValueError, match=r"unwired runtime deps.*required_dep"):
            check_wired(comp)

    def test_ignores_optional_deps(self):
        comp = _Component(required_dep="wired")
        # optional_dep has a default, so it's fine even without explicit wiring
        check_wired(comp)

    def test_label_appears_in_error(self):
        comp = _Component()
        with pytest.raises(ValueError, match=r"estimators\['kinetic'\]"):
            check_wired(comp, label="estimators['kinetic']")

    def test_no_deps_passes(self):
        comp = _NoDeps()
        check_wired(comp)  # should not raise

    def test_non_dataclass_is_noop(self):
        check_wired("not a dataclass")  # should not raise
        check_wired(42)  # should not raise

    def test_unset_sentinel_repr(self):
        comp = _Component()
        assert comp.required_dep is UNSET
        assert repr(UNSET) == "<UNSET>"
