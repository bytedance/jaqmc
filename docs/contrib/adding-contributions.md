# Adding a contribution

This guide explains how to add an independently maintained package under
`contrib/`.

## 1. Create the package layout

```text
contrib/<short-name>/
  pyproject.toml
  README.md
  src/jaqmc_contrib_<short_name>/
    __init__.py
    cli.py  # when providing a CLI
  tests/
  docs/index.md
```

Use `<short-name>` for the package directory and distribution suffix, and use
`<short_name>` for the corresponding Python import package. For example,
`my-feature` becomes `contrib/my-feature/`, `jaqmc-contrib-my-feature`, and
`jaqmc_contrib_my_feature`.

## 2. Add the JaQMC dependency

In `contrib/<short-name>/pyproject.toml`:

```toml
[project]
name = "jaqmc-contrib-my-feature"
version = "0.1.0"
description = "My JaQMC contribution"
license = "Apache-2.0"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "jaqmc",
]

[tool.uv.sources]
jaqmc = { workspace = true }

[build-system]
requires = ["uv_build>=0.11.0,<0.12.0"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "jaqmc_contrib_my_feature"
```

This makes the contribution depend on the workspace copy of JaQMC. Add other
direct dependencies that the package uses.

## 3. Add an optional CLI

Add Click to the existing `[project].dependencies` in
`contrib/<short-name>/pyproject.toml`:

```toml
dependencies = [
    "click",
    "jaqmc",
]
```

Then create `src/jaqmc_contrib_<short_name>/cli.py`. A contribution CLI must
expose a `click.Command` or `click.Group`; the following minimal group keeps
JaQMC imports inside its command callback:

```python
import click


@click.group(help="My contribution commands.")
def cli() -> None:
    pass


@cli.command(help="Show contribution information.")
def info() -> None:
    from jaqmc.utils.config import ConfigManager

    config = ConfigManager({"package_name": "jaqmc-contrib-my-feature"})
    click.echo(f"configured package: {config.get('package_name', '')}")
```

This follows the core CLI pattern: once Click resolves a command, its callback
imports the workflow, optional backend, or other heavyweight code it needs.
Keep the entry-point module limited to Click and small standard-library imports.

Then register the completed `cli` object with a unique `jaqmc.apps` name that
does not collide with a built-in command:

```toml
[project.entry-points."jaqmc.apps"]
my-feature = "jaqmc_contrib_my_feature.cli:cli"
```

JaQMC does not depend on contribution packages. It reads provider names from
installed package metadata without importing them; when the user invokes a
contributed command, Click imports that contribution's CLI module. The command
callback then loads its heavyweight dependencies only when that subcommand runs.

## 4. Install and run a contribution

The root `uv sync` installs JaQMC and its dependencies, but not the
contribution packages in the workspace. To develop or run one contribution,
sync that package explicitly:

```bash
uv sync --package jaqmc-contrib-<short-name>
source .venv/bin/activate
```

This installs the contribution and its regular `[project.dependencies]`. To
install every contribution package, use `uv sync --all-packages`. Use the
`jaqmc.apps` entry-point name as the `jaqmc` command name. For the
`my-feature` example above:

```bash
jaqmc my-feature --help
```

More generally, run a contribution with
`jaqmc <entry-point> ...`.

## 5. Add package-local tests

Place tests in `contrib/<short-name>/tests/`. When run from the contribution
directory, pytest discovers this directory without additional configuration; add
`pyproject.toml` settings only when the package needs them, such as warning
filters.

From the repository root, run:

```bash
uv sync --locked --group test --package jaqmc-contrib-<short-name>
(cd contrib/<short-name> && uv run pytest)
```

CI discovers each workspace contribution and gives it a separate sync, build,
and test matrix job.

## 6. Write package documentation

Add `contrib/<short-name>/docs/index.md` and additional pages as needed. Sphinx
Collections publishes the tree under `contrib/packages/<short-name>/` in the
main site.

## 7. Configure quality gates

Each contribution should include:

- Apache-2.0 licensing in package metadata and headers on Python sources. A
  separate `LICENSE` file in the package directory is not required; the
  repository `LICENSE` covers contributions in this workspace.
- README with support status and install instructions

Repository pre-commit hooks still run license insertion and formatting on
contrib Python files.

## 8. Verify before opening a PR

From the repository root:

```bash
uv lock
uv sync --locked --group test --package jaqmc-contrib-<short-name>
source .venv/bin/activate
(cd contrib/<short-name> && uv run pytest)
uv build --package jaqmc-contrib-<short-name>
jaqmc <entry-point> --help
uv run --group docs sphinx-build -W -b html docs docs/_build/html
uv run pytest tests/contrib_cli_test.py
```

See `contrib/example` for a complete minimal reference implementation.
