# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import yaml

from jaqmc.utils.yaml_format import (
    annotate_yaml_with_sources,
    dump_yaml,
)


class TestDumpYaml:
    def test_numeric_list_flow_style(self):
        data = {"coords": [1.0, 2.0, 3.0]}
        yaml_str = dump_yaml(data)
        assert "coords: [1.0, 2.0, 3.0]" in yaml_str

    def test_matrix_flow_style_rows(self):
        data = {
            "lattice": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        }
        yaml_str = dump_yaml(data)
        assert "- [1.0, 0.0, 0.0]" in yaml_str
        assert "- [0.0, 1.0, 0.0]" in yaml_str
        assert "- [0.0, 0.0, 1.0]" in yaml_str

    def test_mixed_list_block_style(self):
        data = {"mixed": [1, "text", 3]}
        yaml_str = dump_yaml(data)
        # Mixed lists should use block style
        assert "- 1" in yaml_str
        assert "- text" in yaml_str
        assert "- 3" in yaml_str

    def test_empty_list(self):
        data = {"empty": []}
        yaml_str = dump_yaml(data)
        assert "empty: []" in yaml_str

    def test_accepts_plain_dict(self):
        data = {"coords": [1.0, 2.0, 3.0], "name": "test"}
        yaml_str = dump_yaml(data)
        assert "coords: [1.0, 2.0, 3.0]" in yaml_str
        assert "name: test" in yaml_str

    def test_roundtrip_via_yaml(self):
        original = {
            "name": "true",
            "lattice": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "coords": [1.0, 2.0, 3.0],
            "nested": {"value": 42},
        }
        yaml_str = dump_yaml(original)
        loaded = yaml.safe_load(yaml_str)

        assert loaded["name"] == "true"
        assert loaded["coords"] == [1.0, 2.0, 3.0]
        assert loaded["lattice"] == [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        assert loaded["nested"]["value"] == 42

    def test_sort_keys(self):
        data = {"z": 1, "a": 2, "m": 3}
        yaml_str = dump_yaml(data, sort_keys=True)
        lines = [line for line in yaml_str.splitlines() if line.strip()]
        assert lines[0].startswith("a:")
        assert lines[1].startswith("m:")
        assert lines[2].startswith("z:")

    def test_tuple_serialized_as_list(self):
        data = {"spins": (3, 2)}
        yaml_str = dump_yaml(data)
        assert "spins: [3, 2]" in yaml_str


class TestAnnotateYamlWithSources:
    def test_no_source_map_returns_unchanged(self):
        yaml_str = "key: value\n"
        result = annotate_yaml_with_sources(yaml_str, {})
        assert result == yaml_str

    def test_adds_source_comment(self):
        yaml_str = "section:\n  key: value\n"
        source_map = {"section": ("/path/to/file.py", 42, None)}
        result = annotate_yaml_with_sources(yaml_str, source_map)
        assert "# Defined in /path/to/file.py:42" in result

    def test_nested_path_annotation(self):
        yaml_str = "a:\n  b:\n    c: 1\n"
        source_map = {"a.b": ("/file.py", 10, None)}
        result = annotate_yaml_with_sources(yaml_str, source_map)
        lines = result.splitlines()
        # Find the comment and verify it's before 'b:'
        for i, line in enumerate(lines):
            if "b:" in line and i > 0:
                assert "# Defined in /file.py:10" in lines[i - 1]
                break

    def test_verbose_includes_docstring(self):
        yaml_str = "section:\n  key: value\n"
        source_map = {"section": ("/file.py", 42, "My docstring.")}
        result = annotate_yaml_with_sources(yaml_str, source_map, verbose=True)
        assert "# Defined in /file.py:42" in result
        assert "# My docstring." in result

    def test_verbose_false_excludes_docstring(self):
        yaml_str = "section:\n  key: value\n"
        source_map = {"section": ("/file.py", 42, "My docstring.")}
        result = annotate_yaml_with_sources(yaml_str, source_map, verbose=False)
        assert "# Defined in /file.py:42" in result
        assert "# My docstring." not in result

    def test_multiline_docstring(self):
        yaml_str = "section:\n  key: value\n"
        source_map = {"section": ("/file.py", 42, "Line 1.\nLine 2.")}
        result = annotate_yaml_with_sources(yaml_str, source_map, verbose=True)
        assert "# Line 1." in result
        assert "# Line 2." in result
