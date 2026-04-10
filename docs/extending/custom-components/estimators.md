# Custom Estimators

Estimators are the most common extension point — every new observable is a new estimator. This page walks through building one, starting from the simplest approach and adding complexity only when needed.

## Start with a Function

The simplest estimator is a plain function. Here's one that computes the electronic dipole moment:

```python
import jax.numpy as jnp

def dipole_moment(params, data, stats, state, rngs):
    del params, stats, rngs
    dipole = -jnp.sum(data.electrons, axis=0)
    return {"dipole_x": dipole[0], "dipole_y": dipole[1], "dipole_z": dipole[2]}, state
```

Pass it directly in your estimators dict — JaQMC wraps it as a {class}`~jaqmc.estimator.base.FunctionEstimator`:

```python
estimators = {"dipole": dipole_moment, ...}
```

The function receives a single walker's data (not a batch). The base class
auto-vmaps it over walkers. If you need the exact `Data` versus `BatchedData`
convention behind that statement, see <project:../runtime-data-conventions.md>. If
this is all you need — a fixed computation with no tunable parameters — you're
done.

## Making It Configurable

When users should be able to adjust parameters via YAML, wrap the function in a class:

```python
from dataclasses import field

from jaqmc.estimator import Estimator
from jaqmc.utils.config import configurable_dataclass

@configurable_dataclass
class DipoleEstimator(Estimator):
    """Estimates the electric dipole moment.

    Args:
        reference_point: Origin for dipole calculation.
    """
    reference_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def evaluate_local(self, params, data, prev_local_stats, state, rngs):
        del params, prev_local_stats, rngs
        ref = jnp.array(self.reference_point)
        dipole = -jnp.sum(data.electrons - ref, axis=0)
        return {"dipole_x": dipole[0], "dipole_y": dipole[1], "dipole_z": dipole[2]}, state
```

Now `reference_point` is a config field — it appears in `--dry-run` output and users can override it in YAML:

```yaml
train:
  estimators:
    dipole:
      reference_point: [1.0, 0.0, 0.0]
```

## Adding Runtime Dependencies

The electronic dipole above ignores the nuclear contribution. To include it, the estimator needs the nuclear charges and positions — but these come from the molecule specification, not from YAML. They're set by the workflow at startup.

:::{note}
In practice, molecule workflows already include charges and atom positions in the walker data (`data.charges`, `data.atoms`). We use them as runtime deps here to illustrate the pattern — real candidates are things like `f_log_psi` (the wavefunction evaluate function) or `nspins` (the spin configuration), which genuinely live outside the data.
:::

Mark them as `runtime_dep()` so they stay invisible to the config system:

```python
from jaqmc.utils.wiring import runtime_dep

@configurable_dataclass
class DipoleEstimator(Estimator):
    """Estimates the electric dipole moment (electronic + nuclear).

    Args:
        reference_point: Origin for dipole calculation.
        atom_charges: Nuclear charges Z_I (runtime dep).
        atom_positions: Nuclear positions R_I (runtime dep).
    """
    reference_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    atom_charges: Any = runtime_dep()
    atom_positions: Any = runtime_dep()

    def evaluate_local(self, params, data, prev_local_stats, state, rngs):
        del params, prev_local_stats, rngs
        ref = jnp.array(self.reference_point)
        elec = -jnp.sum(data.electrons - ref, axis=0)
        nuc = jnp.sum(self.atom_charges[:, None] * (self.atom_positions - ref), axis=0)
        dipole = elec + nuc
        return {"dipole_x": dipole[0], "dipole_y": dipole[1], "dipole_z": dipole[2]}, state
```

Construct it directly — you provide the dependencies yourself:

```python
est = DipoleEstimator(atom_charges=system.charges, atom_positions=system.positions)
```

Or make it user-configurable via `cfg.get()` and wire the runtime deps separately:

```python
from jaqmc.utils.wiring import wire

est = cfg.get("estimators.dipole", DipoleEstimator())
wire(est, atom_charges=system.charges, atom_positions=system.positions)
```

In YAML, only `reference_point` appears. The runtime deps are invisible:

```yaml
train:
  estimators:
    dipole:
      reference_point: [1.0, 0.0, 0.0]
```

## Choosing Lifecycle Methods

The base class {class}`~jaqmc.estimator.base.Estimator` provides no-op defaults for all methods. Override only what you need:

| Override | When |
|----------|------|
| `evaluate_local` | You're computing a per-walker observable (most estimators). |
| `evaluate_batch` | You need the full batch at once — e.g., histogram accumulation, or your logic doesn't decompose per-walker. Overrides the auto-vmap. |
| `init(self, data, rngs)` | You need to precompute data from runtime deps before the first step — e.g., building index arrays or vmapping a function. Return the initial `state`. |
| `reduce` | The default mean-with-variance isn't appropriate — e.g., you need median, IQR clipping, or custom aggregation. |
| `finalize_stats` | The final observable requires nonlinear combinations of step-level averages — e.g., ratios, products, or gradient assembly. |
| `finalize_state` | You accumulated results in state (e.g., histograms) rather than through per-step statistics. |

For most custom observables, `evaluate_local` alone is sufficient.

Two things to know about the keys you return from `evaluate_local`:

- **Energy prefix**: Keys starting with `energy:` (e.g., `energy:kinetic`, `energy:potential`) are auto-summed by {class}`~jaqmc.estimator.total_energy.TotalEnergy` into `total_energy`, which becomes the VMC optimization target. Use this prefix if your estimator contributes an energy term.
- **Pipeline order matters**: Estimators run in insertion order, and each receives `prev_local_stats` — the local values from all preceding estimators. If your estimator depends on another's output, place it later in the dict.

## End-to-End: Wire Your Estimator into a Real Run

The sections above showed how to define an estimator (function or class) and choose the
right lifecycle hooks. This section puts those pieces into a complete run path: register
the estimator in a workflow, enable it in config, run training/evaluation, then verify the
new stats.

### 1) Register it in the workflow

In your app's workflow module, import your estimator and add it to the estimator collection
behind an explicit config flag. The concrete code below uses
`src/jaqmc/app/molecule/workflow.py` as an example:

```python
from jaqmc.app.molecule.dipole_estimator import DipoleEstimator
```

```python
if cfg.get("estimators.enabled.dipole", False):
    estimators["dipole"] = cfg.get(
        "estimators.dipole",
        DipoleEstimator(),
    )
```

Registering in the shared estimator factory (for molecule, `make_estimators(...)`) makes it
available to both training and evaluation when both stages use that factory.

### 2) Enable it in YAML

In your run config:

```yaml
estimators:
  enabled:
    dipole: true
  dipole:
    reference_point: [0.0, 0.0, 0.0]
```

### 3) Run training and evaluation

```bash
# Train
jaqmc molecule train --yml water.yml workflow.save_path=./runs/water-dipole \
  train.run.iterations=200

# Evaluate
jaqmc molecule evaluate --yml water.yml workflow.save_path=./runs/water-dipole-eval \
  workflow.source_path=./runs/water-dipole run.iterations=200
```

### 4) Verify it worked

Quick checks:

1. `train_stats.csv` contains `dipole_x`, `dipole_y`, `dipole_z` columns.
2. `evaluation_stats.h5` contains the same keys.
3. `evaluation_digest.npz` typically includes dipole summary values for the same keys.

If those keys are present and change smoothly over iterations, your estimator is wired and
running correctly.

### Common Pitfalls in This Flow

- **Forgetting workflow registration**: Defining the estimator class alone does nothing; it
  must be inserted into `make_estimators(...)`.
- **Using the `energy:` prefix by accident**: Keys like `energy:dipole` would be folded into
  `total_energy`. For non-energy observables, use plain keys like `dipole_x`.
- **Returning batched values from `evaluate_local`**: `evaluate_local` is single-walker.
  Return per-walker scalars/arrays; batching is handled by the base class.
- **Runtime dependency not wired**: If you use `runtime_dep()`, ensure the workflow calls
  `wire(...)` before the estimator is used.

## See Also

- <project:/guide/estimators/index.md> — the full evaluation lifecycle
- <project:/api-reference/estimators.md> — base class and built-in estimator API
