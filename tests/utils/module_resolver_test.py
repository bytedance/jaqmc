# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import tempfile
import uuid
from pathlib import Path
from types import ModuleType

import pytest

from jaqmc.utils.module_resolver import (
    ModuleResolutionError,
    import_module_or_file,
    resolve_object,
)


# Test fixtures: create mock modules dynamically
def make_module_with_all(monkeypatch, name: str, exports: list[str]) -> ModuleType:
    """Create a module with __all__ defined.

    Returns:
        A dummy module with __all__ attribute and exported objects.
    """
    module = ModuleType(name)
    module.__all__ = exports  # type: ignore[attr-defined]
    # Add some dummy objects
    for export in exports:
        setattr(module, export, f"object_{export}")
    monkeypatch.setitem(sys.modules, name, module)
    return module


def make_module_without_all(monkeypatch, name: str) -> ModuleType:
    """Create a module without __all__.

    Returns:
        A dummy module without __all__ attribute.
    """
    module = ModuleType(name)
    module.some_function = "some_function_obj"  # type: ignore[attr-defined]
    module.another_object = "another_object_obj"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    return module


class TestResolveObject:
    """Tests for resolve_object function."""

    def test_explicit_notation_with_colon(self, monkeypatch):
        """Test explicit 'module:name' notation."""
        # Setup: create a test module
        make_module_with_all(
            monkeypatch, "test_module_explicit", ["target_func", "other"]
        )

        # Test: resolve with explicit notation
        result = resolve_object("test_module_explicit:target_func")

        assert result == "object_target_func"

    def test_shorthand_notation_uses_all_first_item(self, monkeypatch):
        """Test shorthand 'module' notation uses __all__[0]."""
        # Setup: create module with __all__ = ["first", "second"]
        make_module_with_all(monkeypatch, "test_module_shorthand", ["first", "second"])

        # Test: resolve without colon should get first item from __all__
        result = resolve_object("test_module_shorthand")

        assert result == "object_first"  # Should get __all__[0]

    def test_shorthand_without_all_raises_error(self, monkeypatch):
        """Test shorthand notation fails when module has no __all__."""
        # Setup: create module without __all__
        make_module_without_all(monkeypatch, "test_module_no_all")

        # Test: shorthand has no default export
        with pytest.raises(
            ModuleResolutionError, match="does not define a default object"
        ):
            resolve_object("test_module_no_all")

    def test_shorthand_with_empty_all_raises_error(self, monkeypatch):
        """Test shorthand notation fails when __all__ is empty."""
        # Setup: create module with empty __all__
        module = ModuleType("test_module_empty_all")
        module.__all__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "test_module_empty_all", module)

        # Test: shorthand has no default export
        with pytest.raises(
            ModuleResolutionError, match="does not define a default object"
        ):
            resolve_object("test_module_empty_all")

    @pytest.mark.parametrize(
        ("reference", "reason"),
        [
            ("", "module segment is empty"),
            (":object", "module segment before ':' is empty"),
            ("module:", "object segment after ':' is empty"),
            ("module:name:extra", "can contain one colon"),
        ],
    )
    def test_invalid_reference_raises_clear_error(self, reference, reason):
        """Test malformed module references identify the invalid segment."""
        with pytest.raises(ModuleResolutionError, match=reason):
            resolve_object(reference)

    def test_missing_module_reports_import_guidance(self):
        """Test missing modules report environment and spelling guidance."""
        module_name = f"jaqmc_test_missing_{uuid.uuid4().hex}"

        with pytest.raises(
            ModuleResolutionError,
            match=(
                rf"module reference '{module_name}' is well-formed but could not "
                rf"import module '{module_name}': .*Check its spelling"
            ),
        ):
            resolve_object(module_name)

    def test_nonexistent_object_raises_error(self, monkeypatch):
        """Test accessing non-existent object name raises AttributeError."""
        # Setup: create module
        make_module_with_all(monkeypatch, "test_module_missing", ["exists"])

        # Test: accessing a missing object reports its module and object name
        with pytest.raises(
            ModuleResolutionError, match="does not export the requested object"
        ):
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

    def test_package_relative_import(self, monkeypatch):
        """Test package parameter for relative imports."""
        # Setup: create a parent module and submodule
        parent = ModuleType("test_parent")
        monkeypatch.setitem(sys.modules, "test_parent", parent)

        submodule = ModuleType("test_parent.submodule")
        submodule.target = "target_object"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "test_parent.submodule", submodule)

        # Test: resolve with package parameter (relative import)
        result = resolve_object("submodule:target", package="test_parent")

        assert result == "target_object"

    def test_missing_python_file_reports_path_guidance(self, tmp_path):
        """Test missing Python files report path guidance."""
        module_path = tmp_path / "missing_plugin.py"

        with pytest.raises(
            ModuleResolutionError,
            match=(
                rf"could not load Python file '{module_path}': .*Check that the path "
                r"exists"
            ),
        ):
            resolve_object(f"{module_path}:Factory")

    def test_missing_plugin_dependency_is_distinguished(self, tmp_path):
        """Test plugin dependencies are distinguished from the configured module."""
        module_path = tmp_path / "plugin.py"
        missing_dependency = f"jaqmc_test_missing_{uuid.uuid4().hex}"
        module_path.write_text(f"import {missing_dependency}\n", encoding="utf-8")
        loaded_modules = {name for name in sys.modules if name.startswith("jaqmc_")}

        with pytest.raises(
            ModuleResolutionError,
            match=(
                rf"could not import module '{module_path}' because its dependency "
                rf"'{missing_dependency}' is unavailable"
            ),
        ):
            resolve_object(f"{module_path}:Factory")

        assert {name for name in sys.modules if name.startswith("jaqmc_")} == (
            loaded_modules
        )


class TestImportModuleOrFile:
    """Tests for import_module_or_file function."""

    def test_import_regular_module(self, monkeypatch):
        """Test importing a regular Python module."""
        # Setup: create a test module
        module = ModuleType("test_regular_module")
        module.value = 42  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "test_regular_module", module)

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

    def test_import_with_package_fallback(self, monkeypatch):
        """Test package parameter with fallback to absolute import."""
        # Setup: create a module only in absolute namespace
        module = ModuleType("absolute_module")
        module.value = "absolute"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "absolute_module", module)

        # Test: even with package specified, should fallback to absolute import
        result = import_module_or_file("absolute_module", package="nonexistent_package")

        assert result.value == "absolute"

    def test_relative_import_preserves_missing_dependency(self, tmp_path, monkeypatch):
        """Test fallback does not hide a dependency missing inside a package."""
        package_name = f"jaqmc_test_package_{uuid.uuid4().hex}"
        missing_dependency = f"jaqmc_test_missing_{uuid.uuid4().hex}"
        package_path = tmp_path / package_name
        package_path.mkdir()
        (package_path / "__init__.py").write_text("", encoding="utf-8")
        (package_path / "plugin.py").write_text(
            f"import {missing_dependency}\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()

        with pytest.raises(ModuleNotFoundError, match=missing_dependency):
            import_module_or_file("plugin", package=package_name)

        monkeypatch.delitem(sys.modules, package_name, raising=False)
