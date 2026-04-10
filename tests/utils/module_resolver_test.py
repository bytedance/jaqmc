# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

from jaqmc.utils.module_resolver import import_module_or_file, resolve_object


# Test fixtures: create mock modules dynamically
def make_module_with_all(name: str, exports: list[str]) -> ModuleType:
    """Create a module with __all__ defined.

    Returns:
        A dummy module with __all__ attribute and exported objects.
    """
    module = ModuleType(name)
    module.__all__ = exports  # type: ignore[attr-defined]
    # Add some dummy objects
    for export in exports:
        setattr(module, export, f"object_{export}")
    sys.modules[name] = module
    return module


def make_module_without_all(name: str) -> ModuleType:
    """Create a module without __all__.

    Returns:
        A dummy module without __all__ attribute.
    """
    module = ModuleType(name)
    module.some_function = "some_function_obj"  # type: ignore[attr-defined]
    module.another_object = "another_object_obj"  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


class TestResolveObject:
    """Tests for resolve_object function."""

    def test_explicit_notation_with_colon(self):
        """Test explicit 'module:name' notation."""
        # Setup: create a test module
        make_module_with_all("test_module_explicit", ["target_func", "other"])

        # Test: resolve with explicit notation
        result = resolve_object("test_module_explicit:target_func")

        assert result == "object_target_func"

    def test_shorthand_notation_uses_all_first_item(self):
        """Test shorthand 'module' notation uses __all__[0]."""
        # Setup: create module with __all__ = ["first", "second"]
        make_module_with_all("test_module_shorthand", ["first", "second"])

        # Test: resolve without colon should get first item from __all__
        result = resolve_object("test_module_shorthand")

        assert result == "object_first"  # Should get __all__[0]

    def test_shorthand_without_all_raises_error(self):
        """Test shorthand notation fails when module has no __all__."""
        # Setup: create module without __all__
        make_module_without_all("test_module_no_all")

        # Test: should raise ValueError
        with pytest.raises(ValueError, match="Failed to find default object"):
            resolve_object("test_module_no_all")

    def test_shorthand_with_empty_all_raises_error(self):
        """Test shorthand notation fails when __all__ is empty."""
        # Setup: create module with empty __all__
        module = ModuleType("test_module_empty_all")
        module.__all__ = []
        sys.modules["test_module_empty_all"] = module

        # Test: should raise ValueError
        with pytest.raises(ValueError, match="Failed to find default object"):
            resolve_object("test_module_empty_all")

    def test_too_many_colons_raises_error(self):
        """Test that multiple colons in notation raises error."""
        with pytest.raises(ValueError, match="too many colons"):
            resolve_object("module:name:extra")

    def test_nonexistent_object_raises_error(self):
        """Test accessing non-existent object name raises AttributeError."""
        # Setup: create module
        make_module_with_all("test_module_missing", ["exists"])

        # Test: accessing non-existent attribute raises AttributeError
        with pytest.raises(AttributeError):
            resolve_object("test_module_missing:nonexistent")

    def test_real_module_optax_adam(self):
        """Test with real module: optax."""
        try:
            import optax

            # Test: resolve optax's adam optimizer
            result = resolve_object("optax:adam")

            assert result is optax.adam
        except ImportError:
            pytest.skip("optax not available")

    def test_package_relative_import(self):
        """Test package parameter for relative imports."""
        # Setup: create a parent module and submodule
        parent = ModuleType("test_parent")
        sys.modules["test_parent"] = parent

        submodule = ModuleType("test_parent.submodule")
        submodule.target = "target_object"
        sys.modules["test_parent.submodule"] = submodule

        # Test: resolve with package parameter (relative import)
        result = resolve_object("submodule:target", package="test_parent")

        assert result == "target_object"


class TestImportModuleOrFile:
    """Tests for import_module_or_file function."""

    def test_import_regular_module(self):
        """Test importing a regular Python module."""
        # Setup: create a test module
        module = ModuleType("test_regular_module")
        module.value = 42
        sys.modules["test_regular_module"] = module

        # Test: import should work
        result = import_module_or_file("test_regular_module")

        assert result.value == 42

    def test_import_python_file(self):
        """Test importing a .py file from filesystem."""
        # Setup: create a temporary .py file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("test_variable = 'from_file'\n")
            tmp.write("def test_function():\n")
            tmp.write("    return 'hello'\n")
            tmp_path = tmp.name

        try:
            # Test: import the file
            result = import_module_or_file(tmp_path)

            assert result.test_variable == "from_file"
            assert result.test_function() == "hello"
        finally:
            # Cleanup
            Path(tmp_path).unlink()

    def test_import_nonexistent_file_raises_error(self):
        """Test importing non-existent .py file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            import_module_or_file("/nonexistent/path/file.py")

    def test_import_with_package_fallback(self):
        """Test package parameter with fallback to absolute import."""
        # Setup: create a module only in absolute namespace
        module = ModuleType("absolute_module")
        module.value = "absolute"
        sys.modules["absolute_module"] = module

        # Test: even with package specified, should fallback to absolute import
        result = import_module_or_file("absolute_module", package="nonexistent_package")

        assert result.value == "absolute"
