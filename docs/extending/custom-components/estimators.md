# Custom Estimators

Estimators compute observables such as energies, densities, losses, and
diagnostics. Most custom estimators are easiest to write as single-walker
calculations: you describe what to compute for one sampled configuration, and
JaQMC maps that calculation over the walker batch.

The sections below build in order: a plain function, then a configurable
`LocalEstimator`, then runtime dependencies, then the full-batch `Estimator`
when the whole walker batch is required. After that, a short reference section
pulls the options together; the last section shows how to register the
estimator in a real workflow and config.

## Start with a Function

The simplest estimator is a plain function. Here's one that computes the
electronic dipole moment for one walker:

```python
from jax import numpy as jnp


def dipole_moment(params, data, stats, state, rngs):
    del params, stats, rngs
    dipole = -jnp.sum(data.electrons, axis=0)
    return {"dipole_x": dipole[0], "dipole_y": dipole[1], "dipole_z": dipole[2]}, state
```

Pass it directly in your estimators dict:

```python
estimators = {"dipole": dipole_moment, ...}
```

JaQMC wraps the function as a
{class}`~jaqmc.estimator.base.FunctionEstimator`. The function receives one
walker's `Data`, not a batch. The wrapper batches it over walkers during
`evaluate_batch`. If you need the exact `Data` versus `BatchedData` convention,
see <project:../runtime-data-conventions.md>.

This form is a good fit for small, fixed observables. If the estimator needs
user-tunable fields or setup work, use a class instead.

## Make It Configurable

When users should be able to adjust estimator settings from YAML, use
{class}`~jaqmc.estimator.base.LocalEstimator`. It is the base class for
single-walker estimators: you implement `evaluate_local`, and `LocalEstimator`
provides the batched `evaluate_batch` implementation.

```python
from dataclasses import field

from jax import numpy as jnp

from jaqmc.estimator import LocalEstimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class DipoleEstimator(LocalEstimator):
    """Estimates the electric dipole moment.

    Args:
        reference_point: Origin for dipole calculation.
    """

    reference_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def evaluate_local(self, params, data, prev_local_stats, state, rngs):
        del params, prev_local_stats, rngs
        ref = jnp.array(self.reference_point)
        dipole = -jnp.sum(data.electrons - ref, axis=0)
        return {
            "dipole_x": dipole[0],
            "dipole_y": dipole[1],
            "dipole_z": dipole[2],
        }, state
```

Now `reference_point` is a config field. It appears in `--dry-run` output and
users can override it in YAML:

```yaml
train:
  estimators:
    dipole:
      reference_point: [1.0, 0.0, 0.0]
```

Because this estimator inherits from `LocalEstimator`, users also get
`vmap_chunk_size`. The default `null` evaluates the whole walker batch in one
vmap. On memory-limited runs, users can set an integer to evaluate fewer
walkers at a time:

```yaml
train:
  estimators:
    expensive_observable:
      vmap_chunk_size: 128
```

That knob is specific to the local-vmap path. Estimators that implement their
own full-batch `evaluate_batch` do not inherit it, because JaQMC cannot know how
to split their batch logic safely.

## Add Runtime Dependencies

Config fields come from YAML. Runtime dependencies come from the workflow at
startup: wavefunction evaluate functions, system metadata, spin counts, or
other live objects that should not be serialized into config.

For example, the electronic dipole above ignores the nuclear contribution. To
include it, the estimator needs nuclear charges and positions. Mark those fields
with {func}`~jaqmc.utils.wiring.runtime_dep` so the config system hides them.

:::{note}
Molecule workflows usually include charges and atom positions in the walker
data (`data.charges`, `data.atoms`). This example uses them as runtime
dependencies to show the pattern. More common runtime dependencies are values
such as `f_log_psi` or `nspins`.
:::

```python
from dataclasses import field
from typing import Any

from jax import numpy as jnp

from jaqmc.estimator import LocalEstimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class DipoleEstimator(LocalEstimator):
    """Estimates the electric dipole moment.

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
        return {
            "dipole_x": dipole[0],
            "dipole_y": dipole[1],
            "dipole_z": dipole[2],
        }, state
```

Construct it directly when you already have the dependencies:

```python
est = DipoleEstimator(atom_charges=system.charges, atom_positions=system.positions)
```

Or let config create the user-tunable part, then wire runtime dependencies:

```python
from jaqmc.utils.wiring import wire


est = cfg.get("estimators.dipole", DipoleEstimator())
wire(est, atom_charges=system.charges, atom_positions=system.positions)
```

Only `reference_point` and inherited `LocalEstimator` config fields appear in
YAML. Runtime dependencies remain workflow-owned:

```yaml
train:
  estimators:
    dipole:
      reference_point: [1.0, 0.0, 0.0]
```

If a runtime dependency raises an attribute error, check where the estimator is
wired. Values declared with `runtime_dep()` are intentionally absent from YAML,
so the workflow needs to provide them with `wire(...)` before the estimator is
used.

## Use Full-Batch Evaluation When Needed

The patterns above hand each call a single walker's `Data`; JaQMC still applies
those values across the full walker batch. Some observables are more natural
when the implementation can see the leading batch axis in one go: histograms,
pair-correlation accumulators, and stateful evaluation summaries are typical
examples. For those cases, inherit directly from
{class}`~jaqmc.estimator.base.Estimator` and implement `evaluate_batch`.

The method receives
{class}`~jaqmc.data.BatchedData`, so fields listed in
`batched_data.fields_with_batch` already carry the leading walker axis.

```python
from jax import numpy as jnp

from jaqmc.estimator import Estimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class RadiusHistogram(Estimator):
    """Accumulates electron-radius counts over evaluation steps.

    Args:
        bins: Number of histogram bins.
        max_radius: Upper histogram bound.
    """

    bins: int = 100
    max_radius: float = 10.0

    def init(self, data, rngs):
        del data, rngs
        return jnp.zeros(self.bins)

    def evaluate_batch(self, params, batched_data, prev_local_stats, state, rngs):
        del params, prev_local_stats, rngs
        electrons = batched_data.data.electrons
        radii = jnp.linalg.norm(electrons, axis=-1).reshape(-1)
        counts = jnp.histogram(radii, self.bins, (0.0, self.max_radius))[0]
        return {}, state + counts

    def reduce(self, local_stats):
        del local_stats
        return {}

    def finalize_state(self, state, *, n_steps):
        return {"radius_histogram": state, "radius_histogram:n_steps": n_steps}
```

This estimator does not expose `vmap_chunk_size`, because it never calls the
local-vmap implementation. If a full-batch estimator needs a memory-control
option, make that option describe the estimator's own batching strategy.

## Choose the Right Interface

The examples above are the three main patterns. The tables below are a quick
way to pick an interface, see which lifecycle hooks you might override, and
review two naming and ordering rules for the keys you publish and for how
estimators see each other's outputs.

### Which interface?

The choice comes down to what shape of data the estimator really needs:

| Need | Use | Implement |
|------|-----|-----------|
| Fixed single-walker calculation | Plain function | Function body |
| Configurable single-walker calculation | `LocalEstimator` | `evaluate_local` |
| Full-batch or state-accumulating calculation | `Estimator` | `evaluate_batch` |

### Lifecycle hooks

After you choose the interface, override only the methods your estimator needs:

| Method | When to override |
|--------|------------------|
| `init(self, data, rngs)` | You need derived state before the first step, such as precomputed index arrays. |
| `evaluate_local` | You inherit from `LocalEstimator` and compute one walker's local values. |
| `evaluate_batch` | You inherit from `Estimator` and need the whole walker batch at once. |
| `reduce` | The default mean-with-variance is not appropriate. |
| `finalize_stats` | Final values require nonlinear combinations of step-level statistics. |
| `finalize_state` | Final values come from accumulated estimator state. |

For most custom observables, `LocalEstimator.evaluate_local` is the only method
you need to implement.

### Keys and run order

Two conventions matter for the keys you return and for cross-estimator use:

- **Energy prefix**: Keys starting with `energy:` (for example,
  `energy:kinetic` or `energy:potential`) are auto-summed by
  {class}`~jaqmc.estimator.total_energy.TotalEnergy` into `total_energy`,
  which becomes the VMC optimization target. Use this prefix only for real
  energy terms; a non-energy key such as `energy:dipole` will change the
  optimization target.
- **Pipeline order**: Estimators run in insertion order. Each estimator receives
  `prev_local_stats`, the local values from preceding estimators. If your
  estimator depends on another estimator's output, place it later in the
  estimator dict.

## End-to-End: Wire Your Estimator into a Real Run

With the class or function in place and the right interface selected, the
remaining work is app-specific: register the estimator with the workflow, turn
it on in YAML, run training or evaluation, and confirm the new statistics in
the output files.

### 1. Register it in the workflow

In your app's workflow module, import your estimator and add it to the estimator
collection behind an explicit config flag. The concrete code below uses
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

Registering in the shared estimator factory, such as molecule
`make_estimators(...)`, makes it available to every stage that uses that
factory. If your estimator does not appear in a run, check this registration
step first; defining the class only makes it available to import.

### 2. Enable it in YAML

In your run config:

```yaml
estimators:
  enabled:
    dipole: true
  dipole:
    reference_point: [0.0, 0.0, 0.0]
```

### 3. Run training and evaluation

```bash
# Train
jaqmc molecule train --yml water.yml workflow.save_path=./runs/water-dipole \
  train.run.iterations=200

# Evaluate
jaqmc molecule evaluate --yml water.yml workflow.save_path=./runs/water-dipole-eval \
  workflow.source_path=./runs/water-dipole run.iterations=200
```

### 4. Verify it worked

Quick checks:

1. `train_stats.csv` contains `dipole_x`, `dipole_y`, `dipole_z` columns.
2. `evaluation_stats.h5` contains the same keys.
3. `evaluation_digest.npz` includes the expected final values when the
   estimator finalizes stats or state.

If those keys are present and change smoothly over iterations, your estimator is
wired into the run.

## See Also

- <project:/guide/estimators/index.md> — the full evaluation lifecycle
- <project:/api-reference/estimators.md> — base class and built-in estimator API
