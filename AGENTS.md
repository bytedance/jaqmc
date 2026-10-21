# JaQMC: Agent Guide

## Project Background

**JaQMC** is a JAX-based framework for neural network Quantum Monte Carlo (QMC) and variational wavefunction optimization. It uses deep neural networks as variational wavefunctions to solve the electronic Schrödinger equation without relying on basis sets or density functionals.

Each calculation composes modular, swappable components: `workflow + wavefunction + sampler + estimators + optimizer + writers + config`.

**Stack**: Python >=3.12, JAX, Flax (Linen), Optax, KFAC-JAX, PySCF. Package manager: `uv`.

## Directory Layout & File Placement

Use this structure to locate existing code and place new files where they belong architecturally:

```text
├── src/jaqmc/
│   ├── app/          # System-specific CLI apps, presets, Hamiltonians, and ansatzes (molecule, solid, hall, moire, electron_gas, hydrogen_atom)
│   ├── workflow/     # Generic workflows (VMC, evaluation) and WorkStage execution loops
│   ├── wavefunction/ # Reusable neural wavefunction architectures, backbones (FermiNet, LapNet, Psiformer), inputs/outputs, Jastrows
│   ├── estimator/    # Observable & energy estimators (kinetic, Ewald, ECP, pseudopotential, density, angular momentum, loss gradients)
│   ├── laplacian/    # Forward Laplacian custom JAX transformation and primitive rules
│   ├── optimizer/    # Optimizers & natural-gradient preconditioners (KFAC, SR/JASR, Optax wrappers, schedules)
│   ├── sampler/      # MCMC samplers
│   ├── geometry/     # Coordinate systems and boundary conditions (OBC, PBC, spherical)
│   ├── writer/       # Output sinks (console, HDF5, CSV, W&B)
│   └── utils/        # Cross-cutting utilities (config/CLI resolution, checkpointing, JAX parallelism, atomic/PySCF helpers, units)
├── contrib/          # Standalone or third-party extensions (`jaqmc-contrib-*`) with separate dependencies
├── tests/            # Unit tests in `tests/<subpackage>/*_test.py` mirroring `src/jaqmc/`, plus end-to-end/multi-device tests in `tests/integration/`
├── docs/             # Sphinx documentation (MyST Markdown)
└── src/jaqmc_legacy/, tests/legacy/, legacy_examples/ # Frozen pre-0.1 legacy namespace (never add new features or files here)
```

- **Reusable vs. system-specific**: Put components reusable across physical systems in their domain package (`wavefunction/`, `estimator/`, `geometry/`, `optimizer/`, `sampler/`, `writer/`), and keep single-system assembly under `src/jaqmc/app/<system>/`.
- **Core vs. `contrib/`**: Put experimental or third-party extensions in `contrib/<short-name>/` rather than expanding core `src/jaqmc/` dependencies.

## Code Architecture & Cohesion

- **Cohesive Design (No Patchwork)**: Implementations must feel native to the existing codebase rather than an accumulation of ad-hoc workarounds, special-case flags, or layered hotfixes. Follow existing protocols, stage builders, and configuration patterns:
  - **Config & Wiring**: Decorate configurable components with `@configurable_dataclass` (`jaqmc.utils.config`) and declare runtime-injected dependencies with `runtime_dep()` / `wire()` (`jaqmc.utils.wiring`).
  - **Workflows & Stages**: Assemble training and evaluation pipelines through `Workflow` (`jaqmc.workflow.base`), work stages (`VMCWorkStage`, `EvaluationWorkStage`, `SamplingWorkStage`), and stage builders (`VMCStageBuilder`, `EvalStageBuilder`, `SamplingStageBuilder` in `jaqmc.workflow.stage`).
  - **Component Protocols**: Subclass or implement the domain base classes/protocols (`Wavefunction` / `WavefunctionLike` in `jaqmc.wavefunction.base`, `Estimator` / `PerWalkerEstimator` in `jaqmc.estimator.base`, `SamplerLike` in `jaqmc.sampler.base` / `MCMCSampler` in `jaqmc.sampler.mcmc`, `Writer` in `jaqmc.writer.base`).
- **Simplicity & Directness**: Favor straightforward, minimal implementations over premature abstractions or gratuitous indirection. Avoid single-use, trivial helper functions that obscure control flow.
- **Numerical Stability (typically `float32`)**: Most calculations will run in `float32` on GPUs. Keep numerical stability in mind (e.g., work in the log domain and guard against singularities or catastrophic cancellation).

## Testing

### Running Tests

```bash
uv run pytest tests/utils/config_test.py  # Run targeted test file(s) (preferred during iteration)
uv run pytest                             # Run full suite sequentially
uv run pytest -n <workers>                # Run in parallel with an explicit worker count
uv run jaqmc <app> train --dry-run        # Validate config resolution and workflow wiring without running training
```

> **Note on parallelization**: Do not pass `-n auto` unless `PYTEST_XDIST_AUTO_NUM_WORKERS` is set. By default, `pytest-xdist` reads the host CPU count rather than container or cgroup CPU/memory limits, which can spawn too many JAX worker processes and exhaust memory. Prefer targeted test runs during development, and use an explicit, conservative `-n <workers>` count when running the full suite.

### Writing Tests Wisely

- **Strong, Meaningful Assertions**: Write assertions that validate actual outputs, physical invariants (e.g., antisymmetry, exact analytic limits, shape/JIT compatibility), and side effects. Avoid tautologies or smoke-only checks that merely verify code runs without error, and cover critical edge cases and failure modes.
- **No Duplicate or Redundant Tests**: Do not add multiple tests that exercise identical execution paths with negligible variations. Use `@pytest.mark.parametrize` cleanly when testing distinct cases, and avoid redundant coverage that adds maintenance overhead.
- **Test Public Interfaces, Not Private Internals**: Target public APIs and observable behavior rather than private methods (`_method`) or internal state. Tests coupled to internal mechanics make refactoring brittle.
- **Proportionality (Avoid Over-Testing)**: Keep test suite size proportional to feature complexity and failure impact. Do not write bloated test files for small, low-risk changes.

## Linting, Type Checking, and Pre-commit

We enforce code quality using **Ruff** (linting, formatting, Google-style docstrings, and Apache-2.0 license headers), **Mypy** (type checking), and **prek** (pre-commit runner):

```bash
uv run ruff check . && uv run ruff format .  # Lint and auto-format
uv run mypy .                                # Type check
uvx prek run --all-files                     # Run all pre-commit hooks (ruff, mypy, license headers, pyproject/lock checks)
```

- Fix Ruff and Mypy issues at their root cause rather than suppressing them with `# noqa` or `# type: ignore`, unless strictly unavoidable or following an established codebase pattern for that specific case.
- Write accurate Python 3.12+ type annotations for all new code (avoid `from __future__ import annotations`).

## Documentation

### Standards

- **Completeness & Quality**: User-facing features, CLI/configuration toggles, and behavioral changes must include synchronized documentation updates under `docs/` (or `contrib/<short-name>/docs/` for contrib packages, which Sphinx Collections copies into `docs/contrib/packages/<short-name>/` at build time):
  - **Placement & Flow**: Put concepts in their natural home (link instead of duplicating), and show concrete usage before abstraction.
  - **Seamless Integration**: Fold revisions into the surrounding narrative so the page reads as originally written that way, rather than as appended patch notes or "note that" caveats.
  - **Tone & Style**: Prefer capability-focused phrasing ("X computes...", "Behavior is configured by...") over imperatives ("Use X to...") outside step-by-step task flows, and avoid em dashes or semicolons in prose.
  - **Cross-References**: Use `<project:path/to/page.md>` or `<project:#target>` for automatic titles, and `[custom text](path/to/page.md)` or `[custom text](#target)` for custom link text. Explicit `(target)=` labels are global across the docset, so reference them directly as `#target` rather than `path/to/page.md#target`. For Python symbols, use `{class}`, `{func}`, `{meth}`, or `{mod}` roles (e.g., ``{class}`~jaqmc.wavefunction.Wavefunction` ``). For configuration keys rendered by `config-defaults`, use the `{cfgkey}` role (e.g., ``{cfgkey}`workflow.batch_size <systems-molecule-train-cfg-workflow-batch-size>` ``). For repository source files and directories, prefer the `{ghsrc}` role (e.g., ``{ghsrc}`src/jaqmc/app/hydrogen_atom.py` ``).
  - **Math Formatting**: In MyST Markdown pages (`docs/**/*.md`), use dollar math (`$...$` for inline and `$$...$$` for display blocks). In Python docstrings (which are parsed as reST), use `:math:` and `.. math::`.
- **Concise Comments & Docstrings**: Write Google-style docstrings (using reST syntax, such as `` ``code`` `` and `:math:` / `.. math::`) and inline comments that are accurate and informative without restating obvious code mechanics. Document config dataclass fields under `Args:` in the class docstring so docs can extract them.

### Building Documentation

```bash
uv run --group docs sphinx-build -W -b html docs docs/_build/html   # Strict build (treats warnings as errors, matches CI)
uv run --group docs sphinx-autobuild docs docs/_build --watch src   # Live-reload preview server at http://localhost:8000
```

## Commits

- **Subject line**: Lowercase (unless starting with a proper noun), **max 50 characters**.
- **Body**: Explain *why* the change was made and what collaborators should be aware of going forward, rather than listing what changed.
