<p align="center"><img src="docs/_static/jaqmc-light.svg" width="200" height="200"></p>
<h1 align="center">JaQMC</h1>
<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg"></a>
</p>


**JaQMC** is a JAX-based framework for neural network quantum Monte Carlo (QMC).
It uses deep neural networks as variational wavefunctions to solve the electronic
Schrödinger equation, achieving high accuracy without relying on basis sets or
density functionals.

If you’re looking for the **old (pre-0.1) JaQMC codebase / APIs**, use the legacy
namespace: `import jaqmc_legacy`.

If you're interested in neural network QMC and want to get started quickly,
run real calculations, or build on top of a clean codebase — JaQMC is designed
for you.

## Why JaQMC

- **Modular by design.** Wavefunctions, samplers, estimators, and optimizers are
  independent, swappable components. Want to try a different architecture or loss
  function? Swap one piece without rewriting the rest.
- **General-purpose.** Built to model interacting electrons in any setting —
  atoms, molecules, crystals, and beyond. Ships today with molecular,
  solid-state, and fractional quantum Hall (FQHE) support, with an architecture
  that extends naturally to other quantum systems.
- **Ready to use, ready to modify.** Ships with FermiNet and PsiFormer
  architectures, KFAC, SR, and Optax-based optimizers (for example Adam), and
  preset configurations for common systems. Everything is configurable from the
  CLI — or from code if you want deeper control.
- **Built on JAX.** Automatic differentiation, JIT compilation, and multi-device
  parallelism come for free.

## Installation

Clone the Repository and navigate to the jaqmc directory.
```bash
git clone https://github.com/bytedance/jaqmc.git
cd jaqmc
```

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended):
```bash
uv sync --python 3.12
```

Or with pip (create a venv first):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
```

### GPU Support

For GPU acceleration, choose based on your setup:

```bash
# Option 1: Download CUDA libraries (recommended, no system CUDA needed)
uv sync --extra cuda12
# Or with pip:
pip install -e ".[cuda12]" -r requirements.txt

# Option 2: Use local CUDA installation (requires CUDA 12 already installed)
uv sync --extra cuda12-local
# Or with pip:
pip install -e ".[cuda12_local]" -r requirements.txt
```

For troubleshooting GPU setup, see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Quick Start

```bash
source .venv/bin/activate

# Run the hydrogen atom example
jaqmc hydrogen-atom train

# Customize parameters
jaqmc hydrogen-atom train train.optim.learning_rate.rate=0.01 train.run.iterations=200

# Run a lithium atom simulation
jaqmc molecule train system.module=atom system.symbol=Li
```

See the [documentation](docs/getting-started/quick-start.md) for detailed guides on installation, molecular simulations, and writing custom workflows.

## Where to Go Next

- **Train a real system**: See [Molecules](docs/systems/molecule/index.md), [Solids](docs/systems/solid/index.md), or [Quantum Hall](docs/systems/hall/index.md).
- **Understand the framework**: Read [Core Concepts](docs/getting-started/concepts.md).
- **Extend JaQMC**: Start from [Extending JaQMC](docs/extending/index.md).
- **Contribute code**: See [CONTRIBUTING.md](CONTRIBUTING.md).

## Development

```bash
uv sync
uv tool install prek
prek install

# Run tests
pytest

# Linting and formatting
ruff check .
mypy .
ruff format .
```

## Citation

If you use JaQMC in your research, please cite:

```bibtex
@software{jaqmc2025,
  title = {JaQMC: JAX-based Quantum Monte Carlo},
  author = {ByteDance Seed},
  year = {2025},
  url = {https://github.com/bytedance/jaqmc}
}
```
