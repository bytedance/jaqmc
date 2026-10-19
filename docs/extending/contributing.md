# Contributing

This page covers the development setup, code and docstring conventions, testing options, and local documentation preview for JaQMC.

## Development Setup

Clone the repository and install the locked development environment with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/bytedance/jaqmc.git
cd jaqmc
uv sync --frozen --python 3.12 --extra cuda12   # CPU-only: omit --extra cuda12
source .venv/bin/activate
```

`uv sync --frozen` installs the dependency versions locked in `uv.lock`.

### Pre-commit Hooks

[prek](https://github.com/j178/prek) runs Ruff, Mypy, license header checks, and `requirements.txt` export on each commit.

```bash
uv tool install prek
prek install
```

When a hook reformats a file or inserts a missing license header, stage the updated files with `git add` and commit again.

Those checks can also be run manually from the command line.

```bash
ruff check . && ruff format .   # Lint and format
mypy .                          # Type checking
prek run --all-files            # All pre-commit hooks
```

### VS Code Setup

In VS Code, the [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker) extensions surface lint and type issues inline. For the Mypy extension to resolve project dependencies from `.venv`, set `mypy-type-checker.importStrategy` to `fromEnvironment` in VS Code settings.

## Code Style

[Ruff](https://docs.astral.sh/ruff/) enforces formatting, import sorting, and lint rules, while [Mypy](https://mypy.readthedocs.io/) checks static types.

- **License headers.** Every Python file begins with the Apache-2.0 header. The pre-commit hook inserts this header automatically on new `.py` files.
- **Lint rules.** Ruff errors should be resolved in code rather than suppressed with `# noqa` comments.
- **Type annotations.** New functions, methods, and dataclass fields require standard Python 3.12+ type annotations without `from __future__ import annotations`.

## Docstrings

Docstrings use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) with [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) (reST) markup, which Sphinx renders into the API reference. In reST, inline code uses double backticks (` ``like this`` `) rather than single backticks.

### Math and Citations

Inline math uses the `:math:` role, and displayed equations use the `.. math::` directive.

```rst
The loss :math:`\mathcal{L}` is minimized with respect to parameters :math:`\theta`.

.. math::
    \mathcal{F}_{ij} = \mathbb{E}_{p(\mathbf{X})} \left[
        \frac{\partial \log p(\mathbf{X})}{\partial \theta_i}
        \frac{\partial \log p(\mathbf{X})}{\partial \theta_j}
    \right].
```

Literature references include the journal, volume, page, year, and a DOI link in reST external-link syntax.

```rst
Pfau et al., "Ab initio solution of the many-electron Schrödinger equation
with deep neural networks",
`Phys. Rev. Research 2, 033429 (2020) <https://doi.org/10.1103/PhysRevResearch.2.033429>`_
```

### Config Dataclasses

For classes decorated with `@configurable_dataclass` (and `Wavefunction` subclasses), field descriptions belong in the class docstring under `Args:`. Sphinx extracts field descriptions from `Args:` for the documentation's configuration tables, and `workflow.config.verbose=true` includes the class docstring in resolved configuration output.

## Testing

`pytest` runs both the test suite under `tests/` and doctests in `src/jaqmc/`.

```bash
pytest                          # Full suite
pytest -n 4                     # Parallel via pytest-xdist (adjust worker count for available RAM)
pytest path/to/test_file.py     # Single file
```

Tests focus on observable behavior and public contracts so internal refactors do not break unrelated checks.

`tests/conftest.py` also provides project-specific options and markers.

- **Simulated CPU devices.** The test suite defaults to 2 simulated CPU devices (`--n-cpu-devices=2`) so `pmap` and sharding paths run on single-CPU machines. Passing `--n-cpu-devices=4` overrides the simulated device count.
- **Precision markers.** The `@pytest.mark.x64_modes` marker runs a test under both `float32` and `float64`, while `@pytest.mark.requires_x64` enables double precision for tests that require it.

## Previewing Documentation

The documentation is built with Sphinx and MyST-NB.

```bash
uv sync --frozen --group docs
sphinx-autobuild docs docs/_build --watch src
```

`sphinx-autobuild` serves the site at `http://localhost:8000` and reloads automatically when files under `docs/` or `src/` change.

## Pull Requests

Pull requests should have focused commits and a description that explains *why* the change is needed, with all pre-commit hooks and relevant tests passing.
