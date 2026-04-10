#!/usr/bin/env python3
# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Export design.svg pages to various formats using Inkscape."""

import subprocess
import sys
from pathlib import Path

STATIC_PATH = Path(__file__).parent
EXPORTS = [
    {"page": 1, "format": "svg", "output": "jaqmc-light.svg"},
    {"page": 2, "format": "svg", "output": "jaqmc-light-large.svg"},
    {"page": 5, "format": "svg", "output": "jaqmc-white-large.svg"},
    {"page": 2, "format": "pdf", "output": "jaqmc-light-large.pdf"},
]

SOURCE = STATIC_PATH / "design.svg"


def main():
    for export in EXPORTS:
        cmd = [
            "inkscape",
            f"--export-page={export['page']}",
            "--export-plain-svg" if export["format"] == "svg" else "",
            "-o",
            str(STATIC_PATH / export["output"]),
            str(SOURCE),
        ]
        cmd = [arg for arg in cmd if arg]
        print(f"Exporting {export['output']}...")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Failed to export {export['output']}", file=sys.stderr)
            sys.exit(result.returncode)

    print("All exports complete.")


if __name__ == "__main__":
    main()
