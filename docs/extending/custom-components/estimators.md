# Custom Estimators

Estimators compute observables such as energies, densities, losses, and
diagnostics. Most custom estimators are easiest to write as single-walker
calculations: you describe what to compute for one sampled configuration, and
JaQMC maps that calculation over the walker batch.

The sections below build in order from a configurable `PerWalkerEstimator` to
runtime dependencies and then the full-batch `Estimator` when the whole walker
batch is required. After that, a short reference section pulls the options
together, and the last section shows how to register the estimator in a real
workflow and config.

## Make It Configurable

{class}`~jaqmc.estimator.base.PerWalkerEstimator` is the base class for
single-walker estimators whose settings can be adjusted from YAML. Subclasses
implement `evaluate_single_walker`, and `PerWalkerEstimator` provides the
batched `evaluate_batch_walkers` implementation.

```python
from dataclasses import field

from jax import numpy as jnp

from jaqmc.estimator import PerWalkerEstimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class DipoleEstimator(PerWalkerEstimator):
    """Estimates the electric dipole moment.

    Args:
        reference_point: Origin for dipole calculation.
    """

    reference_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def evaluate_single_walker(
        self, params, data, prev_walker_stats, state, rngs
    ):
        del params, prev_walker_stats, rngs
        ref = jnp.array(self.reference_point)
        dipole = -jnp.sum(data.electrons - ref, axis=0)
        return {
            "dipole_x": dipole[0],
            "dipole_y": dipole[1],
            "dipole_z": dipole[2],
        }, state
```

Now `reference_point` is a config field. It appears in `--dry-run` output and
users can override it in YAML when their workflow declares this estimator:

```yaml
estimators:
  dipole:
    reference_point: [1.0, 0.0, 0.0]
```

Because this estimator inherits from `PerWalkerEstimator`, users also get
`vmap_chunk_size`. The default `null` evaluates the whole walker batch in one
vmap. On memory-limited runs, users can set an integer to evaluate fewer
walkers at a time:

```yaml
estimators:
  expensive_observable:
    vmap_chunk_size: 128
```

That knob is specific to the local-vmap path. Estimators that implement their
own full-batch `evaluate_batch_walkers` do not inherit it, because JaQMC cannot know how
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

from jaqmc.estimator import PerWalkerEstimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class DipoleEstimator(PerWalkerEstimator):
    """Estimates the electric dipole moment.

    Args:
        reference_point: Origin for dipole calculation.
        atom_charges: Nuclear charges Z_I (runtime dep).
        atom_positions: Nuclear positions R_I (runtime dep).
    """

    reference_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    atom_charges: Any = runtime_dep()
    atom_positions: Any = runtime_dep()

    def evaluate_single_walker(
        self, params, data, prev_walker_stats, state, rngs
    ):
        del params, prev_walker_stats, rngs
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

Construct the estimator directly when the nuclear charges and positions are
already available. A molecule configuration exposes them through `atoms`:

```python
from jax import numpy as jnp

atoms = system.atoms
est = DipoleEstimator(
    atom_charges=jnp.asarray([atom.charge for atom in atoms]),
    atom_positions=jnp.asarray([atom.coords for atom in atoms]),
)
```

In a workflow, the estimator is created from configuration first, and the
workflow wires its runtime dependencies in `build_runtime` before passing the
estimators to the stage builder (for example, inside `_wire_estimators` in
{ghsrc}`src/jaqmc/app/molecule/workflow.py`). Wire the estimator even when it is
disabled, so that turning it on later cannot leave its dependencies unset:

```python
from jax import numpy as jnp

from jaqmc.utils.wiring import wire


def _wire_estimators(settings, wf, system, sampler):
    ...
    atoms = system.atoms
    wire(
        settings.dipole,
        atom_charges=jnp.asarray([atom.charge for atom in atoms]),
        atom_positions=jnp.asarray([atom.coords for atom in atoms]),
    )
```

Only `reference_point` and inherited `PerWalkerEstimator` config fields appear in
YAML. Runtime dependencies remain workflow-owned:

```yaml
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
{class}`~jaqmc.estimator.base.Estimator` and implement `evaluate_batch_walkers`.

The method receives
{class}`~jaqmc.data.BatchedData`, so fields listed in
`batched_data.fields_with_batch` already carry the leading walker axis. On
multiple devices, that axis holds only the current device's shard of walkers,
while estimator state is replicated on every device. Any reduction over the
walker axis is therefore local to one device, and a result that feeds the
state needs a cross-device collective from `jaqmc.utils.parallel_jax`,
such as {func}`~jaqmc.utils.parallel_jax.psum` for totals or
{func}`~jaqmc.utils.parallel_jax.pmean` for averages, so the state stays
identical on every device.

```python
from jax import numpy as jnp

from jaqmc.estimator import Estimator
from jaqmc.utils import parallel_jax
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

    def evaluate_batch_walkers(
        self, params, batched_data, prev_walker_stats, state, rngs
    ):
        del params, prev_walker_stats, rngs
        electrons = batched_data.data.electrons
        radii = jnp.linalg.norm(electrons, axis=-1).reshape(-1)
        counts = jnp.histogram(radii, self.bins, (0.0, self.max_radius))[0]
        return {}, state + parallel_jax.psum(counts)

    def reduce(self, walker_stats):
        del walker_stats
        return {}

    def finalize_state(self, state, *, n_steps):
        return {"radius_histogram": state, "radius_histogram:n_steps": n_steps}
```

This estimator does not expose `vmap_chunk_size`, because it never calls the
local-vmap implementation. If a full-batch estimator needs a memory-control
option, make that option describe the estimator's own batching strategy.

## Choose the Right Interface

The examples above are the two main patterns. The tables below are a quick
way to pick an interface, see which lifecycle hooks you might override, and
review two naming and ordering rules for the keys you publish and for how
estimators see each other's outputs.

### Which interface?

The choice comes down to what shape of data the estimator really needs:

| Need | Use | Implement |
|------|-----|-----------|
| Single-walker calculation | `PerWalkerEstimator` | `evaluate_single_walker` |
| Full-batch or state-accumulating calculation | `Estimator` | `evaluate_batch_walkers` |

### Lifecycle hooks

After you choose the interface, override only the methods your estimator needs:

| Method | When to override |
|--------|------------------|
| `init(self, data, rngs)` | You need derived state before the first step, such as precomputed index arrays. |
| `evaluate_single_walker` | You inherit from `PerWalkerEstimator` and compute one walker's values. |
| `evaluate_batch_walkers` | You inherit from `Estimator` and need the whole walker batch at once. |
| `reduce` | The default mean-with-variance is not appropriate. |
| `finalize_stats` | Final values require nonlinear combinations of step-level statistics. |
| `finalize_state` | Final values come from accumulated estimator state. |

For most custom observables, `PerWalkerEstimator.evaluate_single_walker` is the only method
you need to implement.

### Keys and run order

Two conventions matter for the keys you return and for cross-estimator use:

- **Energy prefix**: Keys starting with `energy:` (for example,
  `energy:kinetic` or `energy:potential`) are auto-summed by
  {class}`~jaqmc.estimator.total_energy.TotalEnergy` into `total_energy`,
  which becomes the VMC optimization target. Use this prefix only for real
  energy terms; a non-energy key such as `energy:dipole` will change the
  optimization target.
- **Pipeline order**: Estimators run in the order their fields are declared in
  the workflow's estimator collection; nested collections are expanded in
  place. Each estimator receives `prev_walker_stats`, the per-walker values
  from preceding estimators. If your estimator depends on another estimator's
  output, declare its field after that estimator's field.

## End-to-End: Wire Your Estimator into a Real Run

With the estimator class in place and the right interface selected, the
remaining work is app-specific: register the estimator with the workflow, turn
it on in YAML, run training or evaluation, and confirm the new statistics in
the output files.

### 1. Register it in the workflow

In your app's workflow module, import the estimator from the module where you
defined it and add an optional field to the shared estimator collection.
Adding the field to `MoleculeEstimators` in
{ghsrc}`src/jaqmc/app/molecule/workflow.py` registers it for both training and
evaluation:

```python
from my_package.estimators import DipoleEstimator


@configurable_dataclass
class MoleculeEstimators(EstimatorCollection):
    ...
    dipole: DipoleEstimator = field(default_factory=DipoleEstimator)


@configurable_dataclass
class MoleculeWorkflowBase:
    ...
    estimators: Annotated[
        MoleculeEstimators,
        WithToggles(spin=False, overlap=False, density=False, dipole=False),
    ] = field(default_factory=MoleculeEstimators)
```

Workflows declare which estimators in an
{class}`~jaqmc.estimator.EstimatorCollection` are user-togglable via
{func}`~jaqmc.estimator.WithToggles` (and evaluation workflows omit VMC
`loss_weight` fields via {func}`~jaqmc.estimator.HideLoss`), and
{class}`~jaqmc.estimator.EstimatorPipeline` runs only the enabled estimators
exposed by the collection. Setting `dipole=False` in `WithToggles` exposes an
inline `estimators.dipole.enabled` key defaulting to `false`, while any
defaults on `DipoleEstimator` stay in place.

If the estimator has `runtime_dep` fields, wire them in the workflow's
`build_runtime` method before building the stage, as shown in
[Add Runtime Dependencies](#add-runtime-dependencies).

### 2. Enable it in YAML

Set `estimators.dipole.enabled: true` in YAML to run the estimator:

```yaml
estimators:
  dipole:
    enabled: true
    reference_point: [0.0, 0.0, 0.0]
```

Changing `reference_point` without `estimators.dipole.enabled: true` updates
that setting but leaves the estimator off.

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
2. `evaluation_stats.npy` contains the same keys.
3. `evaluation_digest.npz` includes the expected final values when the
   estimator finalizes stats or state.

If those keys are present and change smoothly over iterations, your estimator is
wired into the run.

## See Also

- <project:/guide/estimators/index.md> — the full evaluation lifecycle
- <project:/api-reference/estimators.md> — base class and built-in estimator API
