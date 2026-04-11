# Quick Start

This guide walks you through installing JaQMC and running your first quantum Monte Carlo simulation — from setup to results in just a few minutes.

## Installation

If you need the old JaQMC APIs, use `jaqmc_legacy`; the rest of this guide
covers the current `jaqmc` package.

Before you begin, make sure you have:

- **Python 3.12** or later
- **Git** (for cloning the repository)

### Step 1: Clone the repository and navigate to the `jaqmc` directory.

```bash
git clone https://github.com/bytedance/jaqmc.git
cd jaqmc
```

### Step 2: Install Dependencies

JaQMC recommends [uv](https://docs.astral.sh/uv/getting-started/installation/) for fast, reliable package management. If you do not have `uv` installed yet, follow the link above.

**Using uv (recommended):**

::::{tab-set}
:sync-group: platform

:::{tab-item} CPU
:sync: cpu
```bash
uv sync --frozen --python 3.12
```
:::

:::{tab-item} GPU (Download CUDA)
:sync: cuda
```bash
uv sync --frozen --python 3.12 --extra cuda12
```
:::

:::{tab-item} GPU (Local CUDA)
:sync: cuda-local
```bash
uv sync --frozen --python 3.12 --extra cuda12-local
```
:::

::::

**Using pip or conda:**

These commands install the exact dependency versions currently tested by the project from the official PyPI index.

If you prefer `pip` or `conda`, create and activate an environment first.

````{dropdown} pip setup
```bash
python -m venv .venv
source .venv/bin/activate
```
````

````{dropdown} conda setup
JaQMC is only tested in standard Python environments. Conda support is provided on a best-effort basis. To avoid dependency conflicts, we strongly recommend against mixing Conda packages with pip packages.

```bash
conda create -n jaqmc python=3.12 -y
conda activate jaqmc
```
````

Then install JaQMC:

::::{tab-set}
:sync-group: platform

:::{tab-item} CPU
:sync: cpu
```bash
pip install -e . -r requirements.txt --extra-index-url https://pypi.org/simple
```
:::

:::{tab-item} GPU (Download CUDA)
:sync: cuda
```bash
pip install -e ".[cuda12]" -r requirements.txt --extra-index-url https://pypi.org/simple
```
:::

:::{tab-item} GPU (Local CUDA)
:sync: cuda-local
```bash
pip install -e ".[cuda12_local]" -r requirements.txt --extra-index-url https://pypi.org/simple
```
:::

::::

The `--extra-index-url https://pypi.org/simple` flag is recommended when you use PyPI mirrors, since some mirrors may not include every required package.

For troubleshooting GPU or JAX issues, see the official [JAX installation guide](inv:jax:*:doc#installation).

### Step 3: Activate the Environment

If you used `uv`, it automatically created a virtual environment in `.venv`. Activate it with:

```bash
source .venv/bin/activate
```

If you used `conda`, make sure your environment is active:

```bash
conda activate jaqmc
```

### Step 4: Verify the Installation

Run the built-in hydrogen atom example to confirm everything is working:

```bash
jaqmc hydrogen-atom train
```

You should see real-time output showing iteration numbers, loss values, energy estimates, and checkpointing activity. If you see this, you're all set!

## Running Your First Simulation

Let's take a closer look at what just happened. The command above ran a variational Monte Carlo (VMC) simulation of a single hydrogen atom — one of the simplest quantum systems, and a great way to verify that JaQMC is working correctly.

The trial wavefunction (ansatz) used in this example is:

$$
\psi_T = \exp(\alpha r)
$$

where $r$ is the distance between the electron and the proton. This form closely resembles the exact ground-state solution $\psi \propto \exp(-r)$. The parameter $\alpha$ starts at $-0.8$, and the optimizer adjusts it toward the exact value of $\alpha = -1.0$. Because the ansatz is well-suited to the physics, convergence is fast.

### Understanding the Training Output

When you run a simulation, you'll see output like this:

```text
I | 12-18 16:53:49 |checkpoint| No checkpoint to restore in: runs/jaqmc_20251218_165349
I | 12-18 16:53:50 |  train   | step=0, pmove=0.52, loss=-0.5046
I | 12-18 16:53:50 |  train   | step=1, pmove=0.53, loss=-0.5016
I | 12-18 16:53:50 |  train   | step=2, pmove=0.54, loss=-0.4983
...
I | 12-18 16:53:50 |  train   | step=99, pmove=0.57, loss=-0.5000
I | 12-18 16:53:50 |checkpoint| Saving checkpoint runs/jaqmc_20251218_165349/train_ckpt_000099.npz
I | 12-18 16:53:50 |  train   | Time per step: 0.003s
```

Here's what the key columns mean:

- **`pmove`** — The MCMC acceptance rate, i.e., the fraction of proposed electron moves that were accepted. Values around 0.5 are typical and healthy.
- **`loss`** — The variational energy estimate. In VMC, this is the mean total energy of the system. Lower is better — for the hydrogen atom, it should converge toward −0.5 Ha (the exact ground-state energy).

In addition to the terminal output, the training workflow saves detailed statistics to `train_stats.csv` and `train_stats.h5` under `workflow.save_path`. If you do not set `workflow.save_path` explicitly, JaQMC creates a timestamped output directory under `runs/`. If the current working directory is inside the source repo, that means the repo-level `runs/` directory; outside the repo, it means `./runs/` in your current working directory. These files live alongside checkpoint files such as `train_ckpt_000099.npz`.

Here's a quick reference for the most important fields in CSV and HDF5:

| Field | What it tells you |
|-------|-------------------|
| `loss` | Variational energy estimate (same as `total_energy`). Lower is better. |
| `pmove` | MCMC acceptance rate. Acceptable range: 0.3–0.7; the sampler auto-tunes toward 0.50–0.55. |
| `total_energy` | Variational energy in Hartree. The main training observable. |
| `total_energy_var` | Variance of the energy estimate. Lower means a better wavefunction. |
| `energy:kinetic` | Kinetic energy component. |
| `energy:potential` | Potential energy component. |

```{important}
Use training logs and `train_stats.*` to monitor optimization, not as the final number you report. For a final energy or other observable, run `jaqmc <app> evaluate` from the trained run and read `evaluation_digest.npz` for the quick summary or `evaluation_stats.h5` for uncertainty analysis. See [Running Workflows](../guide/running-workflows.md) and <project:../guide/analyzing-evaluations.md>.
```

For deeper analysis — convergence plots, error bars, and reporting guidelines — see <project:../guide/training-stats.md>.

```{tip}
You can override any parameter from the command line. Use `--dry-run` to see the full resolved configuration without actually running a simulation. See <project:../guide/configuration.md> for details.
```

## Next Steps

Now that you've run your first simulation, here are some directions to explore:

- **Understand the method** — <project:concepts.md> explains how JaQMC works: the training loop, the four components, and the vocabulary used throughout the docs.
- **Learn the JAX you actually need** — <project:../extending/jax-for-jaqmc.md> maps the small subset of JAX and Flax concepts that show up in JaQMC, with links to the official upstream docs for the full explanations.
- **Simulate a real system** — The <project:../systems/index.md> section has a dedicated guide for each system type: <project:../systems/molecule/index.md>, <project:../systems/solid/index.md>, and <project:../systems/hall/index.md>.
- **Customize your runs** — <project:../guide/configuration.md> covers CLI overrides, YAML files, and inspecting the resolved config.
- **Understand your output** — <project:../guide/running-workflows.md> explains stages, output files, and checkpointing. <project:../guide/training-stats.md>, <project:../guide/analyzing-evaluations.md>, and <project:../guide/analyzing-wavefunctions.md> show how to analyze results.
- **Extend JaQMC** — <project:../extending/index.md> and <project:../extending/writing-workflows.md> cover writing your own components.
