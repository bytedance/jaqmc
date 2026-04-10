# Contributing

This page covers setting up your development environment and the conventions we follow in JaQMC.

## Development Setup

If you plan to contribute code, it's worth setting up a few tools that will catch common issues before they reach code review.

### Pre-commit Hooks

We use [prek](https://github.com/j178/prek) to run Ruff (linting and formatting) and Mypy (type checking) automatically every time you `git commit`. To install the hooks:

```bash
uv tool install prek
prek install
```

If a hook fails, the commit won't go through. Don't worry — most issues (like formatting) are auto-fixed by the hook. Just `git add` the corrected files and commit again.

### VS Code Extensions

If you use VS Code, these two extensions give you real-time feedback as you write:

- **[Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)** — highlights lint issues inline and can auto-fix them on save
- **[Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker)** — shows type errors as you edit, so you catch them before running the code

```{tip}
For Mypy to work correctly with this project's dependencies, we recommend opening your VS Code Settings (Command/Ctrl + `,`), searching for `mypy-type-checker.importStrategy`, and changing it from `useBundled` to `fromEnvironment`.
```

### Running Checks Manually

You can also run the same checks from the command line whenever you like:

```bash
ruff check . && ruff format .   # Lint and format
mypy .                           # Type checking
pytest -n 8                      # Run all tests (~60s)
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [Mypy](https://mypy.readthedocs.io/) for type checking. Both run automatically as [pre-commit hooks](#pre-commit-hooks), so you'll get feedback before each commit.

Please follow these project guidelines:

- **License header** — Every source file needs the Apache-2.0 license header. Copy it from any existing file.
- **Resolve lint errors** — Please fix underlying issues flagged by Ruff rather than suppressing them with `# noqa` comments. If you encounter a linting error that seems incorrect or difficult to resolve, feel free to ask for guidance during code review.
- **Type annotations** — We require type annotations for all new code. They help catch bugs early and make the codebase easier to navigate.

## Docstrings

We write docstrings in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) using [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) (reST) syntax — this is what Sphinx uses to render the documentation.

Note that inline code uses **double** backticks in reST (` ``like this`` `), not single backticks like in Markdown.

### Math

For equations, we use LaTeX wrapped in reST math directives:

**Inline math** — use the `:math:` role:

```rst
The loss :math:`\mathcal{L}` is minimized with respect to parameters :math:`\theta`.
```

**Block math** — use the `.. math::` directive:

```rst
.. math::
    \mathcal{F}_{ij} = \mathbb{E}_{p(\mathbf{X})} \left[
        \frac{\partial \log p(\mathbf{X})}{\partial \theta_i}
        \frac{\partial \log p(\mathbf{X})}{\partial \theta_j}
    \right].
```

### References

When citing papers, please include the journal name, volume, page, year, and a DOI link so readers can find the original. Use either of the following formats:

**Short form:**

```rst
`Phys. Rev. Research 2, 033429 (2020) <https://doi.org/10.1103/PhysRevResearch.2.033429>`_
```

**With authors and title:**

```rst
Pfau et al., "Ab initio solution of the many-electron Schrödinger equation
with deep neural networks",
`Phys. Rev. Research 2, 033429 (2020) <https://doi.org/10.1103/PhysRevResearch.2.033429>`_
```

### Config Dataclasses

If you're writing a config dataclass, document its fields in the **class docstring**
under an `Args:` section (rather than in `__init__`). This ensures verbose config
output can extract field descriptions to display to users.

## Testing

You can run the full test suite in parallel, which usually takes about a minute:

```bash
pytest -n 8                      # Full suite (~60s)
pytest tests/checkpoint_test.py  # Single file
```

When writing tests, try to focus on **observable behavior** — what the code does from the outside — rather than internal implementation details. This keeps tests resilient when we refactor internals.

### Simulating Multiple Devices

You can simulate multiple JAX devices on a single CPU machine:

```bash
pytest --n-cpu-devices=4
```

This sets `XLA_FLAGS=--xla_force_host_platform_device_count=4` before JAX initializes, so every test in the process sees 4 simulated CPU devices. The default is 2. Use this to verify that your code handles sharding and `pmap`/`shard_map` correctly without needing actual GPUs.

For true multi-process tests (e.g. testing distributed training), see `tests/distributed_test.py`, which spawns separate processes via `multiprocessing.Process`.

### Testing a Custom Estimator

Estimator tests should cover the following scenarios:

**1. ``evaluate_local`` returns the right keys and shapes:**

```python
def test_evaluate_local_shape():
    est = MyEstimator()
    data = ...  # single-walker data
    state = est.init(data, jax.random.PRNGKey(0))
    stats, state = est.evaluate_local(
        None, data, {}, state, jax.random.PRNGKey(1)
    )
    assert "my_key" in stats
    assert stats["my_key"].shape == ()  # scalar per walker
```

**2. JIT compatibility** — wrap `evaluate_local` in `jax.jit` and check that it produces finite values. This catches issues with dynamic shapes or Python control flow that JAX cannot trace.

**3. Batched evaluation** — call `evaluate_batch` on a batch of walkers and verify the output has a leading batch dimension:

```python
def test_evaluate_batch():
    est = MyEstimator()
    state = est.init(single_walker_data, jax.random.PRNGKey(0))
    local_stats, state = est.evaluate_batch(
        None, batched_data, {}, state, jax.random.PRNGKey(1)
    )
    assert local_stats["my_key"].shape == (n_walkers,)
```

**4. Physics correctness** — if possible, provide an analytic wavefunction with a known exact value and assert the estimator reproduces it. See `tests/hall_test.py::TestSphericalKinetic` for an example that uses a closed-form Laughlin wavefunction to verify kinetic energy.

### Testing a Custom Wavefunction

When testing custom wavefunctions, ensure these properties are verified:

- **Finite output:** `wf.apply(params, data)` returns finite `logpsi`.
- **Gradient flow:** `jax.grad` of `logpsi` with respect to electron positions is finite.
- **Antisymmetry:** Swapping two same-spin electrons flips the sign of the wavefunction but preserves `logpsi`.
- **Protocol compliance:** `isinstance(wf, MoleculeWavefunction)` (or the relevant protocol) passes.

For a minimal workflow smoke test, build a `ConfigManager` with only 2 pretrain/train iterations and verify the output stats are finite. See `tests/molecule_wavefunction_test.py` for the full pattern.

### Conventions

- **JAX keys:** Use a module-level `TEST_KEY = jax.random.PRNGKey(42)` and always `split` before passing to different calls. Never reuse a key.
- **Tolerances:** Use `pytest.approx(..., rel=2e-5)` for exact agreement. For stochastic results with sampling noise, use wider tolerances (`atol=0.1` or more).
- **Temp files:** Use pytest's built-in `tmp_path` fixture for any test that writes files (checkpoints, stats). This ensures cleanup.
- **Class grouping:** Group tests by concern (`TestEdgeCases`, `TestGradientFlow`) rather than by implementation.
- **Parametrize over implementations:** Use `@pytest.mark.parametrize` to run the same test across multiple implementations (e.g. FermiNet and Psiformer).
- **License header:** All test files need the Apache-2.0 header. Copy it from any existing test file.

## Previewing Documentation

To build and preview the docs locally:

```bash
uv sync --group docs
sphinx-autobuild docs docs/_build --watch src
```

Then visit http://localhost:8000. The server watches for changes to both the docs and source files, so your browser will auto-reload as you edit.

If you'd like a function's docstring to appear in the rendered docs, add an `autofunction` directive to the appropriate page:

````md
```{eval-rst}
.. autofunction:: jaqmc.geometry.pbc.nu_distance
```
````
