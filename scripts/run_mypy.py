# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Run mypy over the repository, excluding untracked files.

prek only stashes unstaged changes to tracked files before running hooks,
so a plain ``mypy .`` would also type-check untracked scratch files that
are not part of the commit. This wrapper discovers untracked Python files
via ``git ls-files --others`` and passes them to mypy as an explicit
``--exclude`` pattern. Gitignored files are covered separately by mypy's
``exclude_gitignore`` setting.
"""

import re
import subprocess
import sys


def main() -> int:
    """Run mypy with untracked files excluded.

    Returns:
        The mypy exit code.
    """
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.py",
            "*.pyi",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    untracked = [path for path in result.stdout.split("\0") if path]

    cmd = [sys.executable, "-m", "mypy", "."]
    if untracked:
        pattern = "|".join(re.escape(path) for path in untracked)
        cmd += ["--exclude", f"(^|/)({pattern})$"]

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
