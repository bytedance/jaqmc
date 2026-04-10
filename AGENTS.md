# JaQMC

IMPORTANT: This document is a living artifact. When making significant architectural or conventional changes, update this file. If you find discrepancies between this document and the codebase, please correct them.

## 1. Project Overview

JaQMC is a JAX-based Quantum Monte Carlo (QMC) framework for variational wavefunction optimization. Each run is defined by composing: `workflow + wavefunction + sampler + estimators + optimizer + config`.

**Stack**: Python 3.12, JAX, Flax (Linen), Optax, KFAC-JAX, PySCF. Package manager: `uv`.

## 2. Architecture

```
src/jaqmc/
├── app/          # CLI entrypoints (hydrogen_atom, molecule, solid, hall)
├── workflow/     # @workflow decorator + WorkStage execution logic
├── wavefunction/ # Flax/Linen neural wavefunctions
├── sampler/      # MCMC samplers
├── geometry/     # OBC and PBC geometry handling
├── estimator/    # Energy estimators (auto-vmapped over walkers)
├── optimizer/    # Optimizer wrappers (Optax, KFAC)
├── writer/       # Output writers (console, HDF5, CSV)
└── utils/        # Config system, CLI, checkpointing, JAX parallelism
```

**Critical pattern** — Estimator two-level evaluate: Implement `evaluate_local` for a **single walker**. The base class `evaluate_batch` auto-vmaps `evaluate_local` over walkers. Override `evaluate_batch` directly for estimators that don't need per-walker vmapping (e.g. histogram aggregation, stats-only estimators).

**Data convention** — Wavefunctions and other single-sample logic usually consume one walker's `Data`. Batched runtime state is represented by `BatchedData`, where `data` keeps the same dataclass structure and `fields_with_batch` marks which fields actually carry the leading walker axis.

## 3. Commands

```bash
uv sync                          # Install deps
pytest -n 8                      # Run all tests (~60s)
pytest tests/checkpoint_test.py  # Run specific test
ruff check . && ruff format .    # Lint + format
mypy .                           # Type checking
prek run --all-files             # All prek hooks (ruff + mypy)
jaqmc hydrogen-atom train --dry-run  # Show resolved config
```

## 4. Documentation Index

Read these files for deeper context on specific topics:

**Getting Started:**
- Quick start guide: `read docs/getting-started/quick-start.md`
- Core concepts (VMC, components, units): `read docs/getting-started/concepts.md`

**Systems** (each system page covers quick start, configuration reference, and advanced topics):
- Molecule simulations: `read docs/systems/molecule/index.md`
- Molecule config reference: `read docs/systems/molecule/train.md`
- Solid simulations: `read docs/systems/solid/index.md`
- Solid config reference: `read docs/systems/solid/train.md`
- Quantum Hall (FQHE): `read docs/systems/hall/index.md`
- Hall config reference: `read docs/systems/hall/train.md`

**Guides:**
- Configuration: `read docs/guide/configuration.md`
- Running workflows: `read docs/guide/running-workflows.md`
- Wavefunctions: `read docs/guide/wavefunction.md`
- How estimators work: `read docs/guide/estimators/index.md`
- Optimizers: `read docs/guide/optimizers/index.md`
- Samplers: `read docs/guide/sampling.md`
- Periodic boundary conditions: `read docs/guide/periodic-boundaries.md`
- Writers: `read docs/guide/writers.md`
- Multi-device training: `read docs/guide/multi-device.md`
- Reading training statistics: `read docs/guide/training-stats.md`
- Analyzing evaluations: `read docs/guide/analyzing-evaluations.md`
- Analyzing wavefunctions: `read docs/guide/analyzing-wavefunctions.md`
- Troubleshooting: `read docs/guide/troubleshooting.md`

**Extending JaQMC:**
- Architecture overview: `read docs/extending/index.md`
- Runtime data conventions: `read docs/extending/runtime-data-conventions.md`
- Writing workflows: `read docs/extending/writing-workflows.md`
- Writing wavefunctions: `read docs/extending/wavefunctions.md`
- Custom components: `read docs/extending/custom-components/index.md`
- Configuration system: `read docs/extending/configuration.md`
- Contributing: `read docs/extending/contributing.md`

**API Reference:**
- Workflows: `read docs/api-reference/workflows.md`
- Stages: `read docs/api-reference/stages.md`
- Configuration API: `read docs/api-reference/configuration.md`
- Wavefunctions: `read docs/api-reference/wavefunctions.md`
- Estimators: `read docs/api-reference/estimators.md`
- Optimizers: `read docs/api-reference/optimizers.md`
- Samplers: `read docs/api-reference/samplers.md`
- Writers: `read docs/api-reference/writers.md`
- Ruff/lint rules: `read pyproject.toml` (section `[tool.ruff]`)

## 5. Style & Constraints

- **JAX imports**: `from jax import numpy as jnp`, never `jax.numpy`.
- **Docstrings**: Google Style, 4-space indented body. Config dataclasses document fields in the **class docstring** `Args:` section (not `__init__`), so `--verbose-config` can extract them.
- **License**: All source files require the Apache-2.0 header.
- **Linting**: Ruff is the single source of truth. Run `ruff check .` before committing. **Never suppress ruff errors** (no `# noqa` for ruff rules, no adding rules to the ignore list). Fix them properly. If you cannot fix a lint error, ask the user.
- **Type checking**: `mypy .` is enforced via prek. Ensure new code has proper type annotations.
- **Testing**: Tests focus on observable behavior. Use `--n-cpu-devices=N` to simulate multi-device.
- **Before committing**: Run `ruff check .`, `ruff format .`, and `mypy .` to ensure lint and type checks pass.
- **Commits**: Subject line lowercase (unless proper noun), **max 50 chars**. Body explains *why* the change was made and what collaborators should be aware of going forward — not a list of what changed.

## 6. Documentation Guidelines

- **Correctness first.** Understand what the code actually does before writing about it. Never guess or infer behavior — read the implementation. If unsure, ask rather than making something up.
- **Describe mechanisms, not vibes.** State what concretely happens and why it matters, not a vague restatement of the feature name.
- **Introduce vocabulary in context.** Use real terminology, but define terms naturally on first use rather than assuming prior knowledge.
- **Structure as a journey.** Guide the reader through a logical progression with transitions, not a disconnected list of facts.
- **One idea per unit.** Each paragraph or section makes one point. Don't bury important information inside a paragraph about something else.

## 7. Scientific Code Guidelines

- **Formulas in docstrings must match the code.** Readers verify correctness by mapping formula terms to code lines — any mismatch destroys trust.
- **Method names should reflect the math**, not just the mechanics. Name methods after what formula term they compute, not what internal operation they perform.
- **Include brief derivations for non-trivial formulas.** Don't just state the final formula — show how it follows from standard definitions so readers can verify the formula itself.
- **Separate implementation tricks from the physics.** Numerical stability tricks, optimization shortcuts, log-space arithmetic — document these separately from the derivation so readers can distinguish "what we compute" from "how we compute it efficiently."
- **No unicode math in inline comments.** Use code-style (`psi`, `sum_j`) or LaTeX (`:math:` in docstrings).
- **Each method should map to a distinct formula term.** Don't collapse methods just to reduce line count if each one corresponds to a meaningful mathematical sub-expression.

## 8. Documentation Review Guidelines

When reviewing documentation pages, apply these checks:

- **Read as the target user.** Review every page from the perspective of someone arriving with a specific goal, not someone who already knows the system. If a concept is used before it's explained or exemplified, that's a problem.
- **Show before explain.** Concrete examples first, then the underlying mechanics. The reader needs to see *what* before understanding *how* or *why*.
- **One source of truth.** Don't duplicate information across pages. Reference material lives in one place; everywhere else links to it. Every duplicate is a maintenance hazard and a reader confusion point.
- **Be consistent and precise.** Examples, diagrams, and prose must tell the same story. Factually incorrect simplifications erode trust faster than complexity does.
- **Earn every structural element.** Every heading, section, and callout box should carry its weight. Orphan paragraphs, one-sentence subsections, and headings that exist "just because" add noise.
