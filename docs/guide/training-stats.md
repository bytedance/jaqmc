---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Training Statistics

This tutorial demonstrates how to read and visualize training statistics from JaQMC computations. You'll learn how to:

1. **Run a computation** and understand the output directory structure
2. **Read statistics from CSV** for quick analysis
3. **Read statistics from HDF5** for full data access
4. **Visualize training progress** with plots

We'll use a simple hydrogen atom as our example system.

````{admonition} Extra Packages
:class: note

This tutorial uses `pandas`, `h5py`, and `matplotlib` for data analysis and plotting.
If you're working from a JaQMC repository clone:
```bash
uv sync --group analysis
```
If you're using JaQMC in your own environment:
```bash
pip install pandas h5py matplotlib
```
````

+++

## Setup

First, let's import the necessary modules. We'll need:
- **pandas** for reading CSV files
- **h5py** for reading HDF5 files
- **matplotlib** for visualization
- **JaQMC modules** for running computations

```{code-cell} ipython3
:tags: [remove-input]

%config InlineBackend.figure_formats = ['svg']
```

```{code-cell} ipython3
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import pandas as pd

from jaqmc.app.molecule import MoleculeTrainWorkflow
from jaqmc.utils.config import ConfigManager
```

We'll use a temporary directory for this tutorial. Any previous runs in the
notebook's execution directory will be cleaned up automatically.

```{code-cell} ipython3
WORKING_DIR = Path("runs/jaqmc_tutorial_stats")

if WORKING_DIR.exists():
    shutil.rmtree(WORKING_DIR)
WORKING_DIR.mkdir(parents=True)
```

## 1. Running a Computation

First, let's run a short VMC training on a hydrogen atom. We'll use a small network and few iterations to keep this tutorial quick. If you prefer the CLI, the equivalent command would look like:

```bash
jaqmc molecule train --yml tutorial-train.yml
```

Here, `tutorial-train.yml` stands for a YAML config with the same settings as the Python `ConfigManager(...)` configuration shown below. Here we use the Python API which is more convenient in notebooks.

```{code-cell} ipython3
:tags: [remove-output]

cfg = ConfigManager(
    {
        "system": {
            "atoms": [{"symbol": "H", "coords": [0.0, 0.0, 0.0]}],
            "electron_spins": [1, 0],
        },
        "workflow": {"batch_size": 256, "save_path": str(WORKING_DIR)},
        "wf": {"hidden_dims_single": [64, 64], "hidden_dims_double": [16, 16]},
        "pretrain": {"run": {"iterations": 100}},
        "train": {"run": {"iterations": 100}},
    }
)

workflow = MoleculeTrainWorkflow(cfg)
workflow()  # Run the workflow
```

## 2. Understanding the Output Directory Structure

After training, JaQMC creates a structured output directory. Let's explore it:

```{code-cell} ipython3
def show_tree(path, prefix=""):
    """Display directory tree structure."""
    path = Path(path)
    contents = sorted(path.iterdir())
    for i, item in enumerate(contents):
        is_last = i == len(contents) - 1
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{item.name}")
        if item.is_dir():
            extension = "    " if is_last else "│   "
            show_tree(item, prefix + extension)

print(f"Output directory: {WORKING_DIR}\n")
show_tree(WORKING_DIR)
```

**Key outputs:**
- `config.yaml` - The resolved configuration used for this run
- `train_ckpt_*.npz` - Checkpoint files containing parameters and walker states
- `train_stats.h5` - HDF5 file with training statistics (energy, variance, etc.)
- `train_stats.csv` - CSV file with the same statistics for easy analysis

+++

## 3. Reading from CSV (Simple)

The CSV file is the easiest way to access training statistics. It can be loaded directly with pandas:

```{code-cell} ipython3
stats_csv = pd.read_csv(WORKING_DIR / "train_stats.csv")
print(f"Training statistics shape: {stats_csv.shape}")
print(f"Available columns: {list(stats_csv.columns)}")
stats_csv.tail()
```

Let's summarize the final training results:

```{code-cell} ipython3
print("Training summary:")
print(f"  Total iterations: {len(stats_csv)}")
print(f"  Final energy: {stats_csv['total_energy'].iloc[-1]:.6f} Ha")
print(f"  Final energy variance: {stats_csv['total_energy_var'].iloc[-1]:.6f} Ha²")
print(f"\n  (Exact H atom: -0.5 Ha)")
```

## 4. Reading from HDF5 (Full Data)

The HDF5 file contains the same data but is more efficient for large datasets and provides direct array access:

```{code-cell} ipython3
with h5py.File(WORKING_DIR / "train_stats.h5", "r") as f:
    print("HDF5 datasets:")
    for key in f.keys():
        print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")

    # Load specific datasets into memory
    loss_history = f["loss"][:]
    total_energy_history = f["total_energy"][:]
```

## 5. Visualizing Training Progress

Let's create some plots to visualize the training convergence.

### Energy and Variance

The total energy should converge toward the exact value (-0.5 Ha for hydrogen). The energy variance measures wavefunction quality - lower variance indicates a better ansatz:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(stats_csv["total_energy"], label="Total Energy")
ax.axhline(y=-0.5, color="r", linestyle="--", label="Exact (-0.5 Ha)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Energy (Ha)")
ax.set_title("Energy Convergence")
ax.legend()

ax = axes[1]
ax.semilogy(stats_csv["total_energy_var"], label="Energy Variance")
ax.set_xlabel("Iteration")
ax.set_ylabel("Variance (Ha²)")
ax.set_title("Energy Variance (lower = better wavefunction)")
ax.legend()

plt.tight_layout()
plt.show()
```

### Energy Components

We can also visualize the kinetic and potential energy separately. For hydrogen, the virial theorem states that $\langle T \rangle = -\langle V \rangle / 2 = 0.5$ Ha at the exact solution:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(stats_csv["energy:kinetic"], label="Kinetic Energy", alpha=0.8)
ax.plot(stats_csv["energy:potential"], label="Potential Energy", alpha=0.8)
ax.plot(stats_csv["total_energy"], label="Total Energy", linewidth=2)
ax.axhline(y=-0.5, color="k", linestyle="--", alpha=0.5, label="Exact Total (-0.5 Ha)")

ax.set_xlabel("Iteration")
ax.set_ylabel("Energy (Ha)")
ax.set_title("Energy Components During Training")
ax.legend()

plt.tight_layout()
plt.show()
```

### MCMC Acceptance Rate

The acceptance rate (`pmove`) measures how often proposed electron moves are accepted. Values in 0.3–0.7 are acceptable; the sampler auto-tunes the step width to keep `pmove` in the 0.50–0.55 target range:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(stats_csv["pmove"], label="Acceptance Rate")
ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Target range")
ax.axhline(y=0.55, color="r", linestyle="--", alpha=0.5)
ax.fill_between(range(len(stats_csv)), 0.5, 0.55, alpha=0.2, color="green", label="Optimal range")

ax.set_xlabel("Iteration")
ax.set_ylabel("Acceptance Rate")
ax.set_title("MCMC Acceptance Rate")
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.show()
```

**Note on acceptance rate:** Short training runs (like this 100-iteration tutorial) often show acceptance rates above the optimal range. Don't worry — JaQMC uses an **adaptive step size strategy** that automatically adjusts the MCMC proposal step size. In longer production runs, the acceptance rate gradually stabilizes within the target range.

+++

## Reporting Energies

When reporting VMC energies (e.g., in a paper), keep the following in mind:

- **Discard the burn-in period.** The first portion of training is far from converged. Only use the final 10–20% of steps for energy estimates.
- **Average over steps.** The mean of `total_energy` over the selected window gives your variational energy estimate.
- **Statistical error.** Each training step's energy is correlated with neighboring steps because the MCMC walkers evolve continuously. A naive standard error (standard deviation / $\sqrt{N}$) will **underestimate** the true uncertainty. For publication-quality error bars, use **block averaging** (also called "blocking analysis"): group consecutive steps into blocks, compute the mean energy per block, and take the standard error of those block means. Increase the block size until the standard error plateaus — that plateau value is the correct statistical uncertainty.
- **Evaluation stage.** For the most reliable final energy, run a separate evaluation (no parameter updates) after training is complete. This avoids any bias from ongoing optimization. See <project:analyzing-evaluations.md> for a full walkthrough.

## Summary

In this tutorial, we learned how to:

1. **Run a JaQMC computation** and locate the output files
2. **Read CSV files** with pandas for quick data exploration
3. **Read HDF5 files** for efficient access to large datasets
4. **Visualize training progress** including energy convergence, variance, and MCMC acceptance rates

For running evaluation with proper error bars, see <project:analyzing-evaluations.md>. For directly evaluating the wavefunction on custom configurations, see <project:analyzing-wavefunctions.md>.
