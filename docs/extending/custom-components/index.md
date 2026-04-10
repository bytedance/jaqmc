# Custom Components

In the <project:../writing-workflows.md> tutorial, we built estimators and passed them to `configure_estimators()`. This works well for simple cases, but as your components grow, you'll want users to be able to tune parameters via YAML — without exposing internal wiring details like `f_log_psi` in the config file.

JaQMC solves this with three building blocks: {deco}`~jaqmc.utils.config.configurable_dataclass`, {func}`~jaqmc.utils.wiring.runtime_dep`, and {func}`~jaqmc.utils.wiring.wire`.

## The Two Kinds of Fields

A typical component has two kinds of fields:

- **Config fields** — User-tunable knobs like `mode`, `cutoff`, or `steps`. These appear in YAML and `--dry-run` output.
- **Runtime dependencies** — Values set programmatically by the workflow at startup, like `f_log_psi` or `nspins`. These don't belong in YAML — they come from the wavefunction, the system config, or other live objects that the workflow controls.

## `@configurable_dataclass`

The {deco}`~jaqmc.utils.config.configurable_dataclass` decorator prepares a class for the config system. It applies `@dataclass` and sets up serialization so that {meth}`~jaqmc.utils.config.ConfigManager.get`, {meth}`~jaqmc.utils.config.ConfigManager.get_module`, and {meth}`~jaqmc.utils.config.ConfigManager.get_collection` can construct instances from YAML.

```python
from jaqmc.utils.config import configurable_dataclass

@configurable_dataclass
class MyEstimator(Estimator):
    cutoff: float = 1e-8  # appears in YAML
```

This is equivalent to writing `@dataclass(kw_only=True)` plus the serialization setup manually, but in one step.

## `runtime_dep()`

Use `runtime_dep()` to declare a field as a runtime dependency. These fields are hidden from serialization — they won't appear in YAML output or be read from config files.

```python
from jaqmc.utils.wiring import runtime_dep

@configurable_dataclass
class MyEstimator(Estimator):
    cutoff: float = 1e-8                              # config field
    f_log_psi: SomeCallable = runtime_dep()            # required
    data_field: str = runtime_dep(default="electrons")  # optional
```

Two forms:

- **`runtime_dep()`** — Required. If accessed before being set, raises a clear error:
  ```
  AttributeError: MyEstimator.f_log_psi is a runtime dependency that was
  not set. Wire it after construction: `instance.f_log_psi = ...`
  or use wire(instance, f_log_psi=...)
  ```

- **`runtime_dep(default=...)`** — Optional. Uses the default if not explicitly wired.

## `wire()`

`wire()` injects runtime dependencies from a keyword dict into a dataclass instance. It sets any `runtime_dep` field whose name matches a key, and raises an error if required deps are still missing.

```python
from jaqmc.utils.wiring import wire

est = MyEstimator()
wire(est, f_log_psi=wf.evaluate)
```

This is equivalent to setting the attribute directly:

```python
est = MyEstimator()
est.f_log_psi = wf.evaluate
```

`wire()` is most useful when you have a context dict with many dependencies and want to inject them all at once — which is exactly what the stage builder does for the sampler, optimizer, and writers it creates.

`wire()` also recurses into nested dataclass fields, so nested components get wired automatically.

## Putting It Together

Here's {class}`~jaqmc.estimator.kinetic.EuclideanKinetic`, a built-in estimator that uses all three mechanisms:

```python
from jaqmc.estimator import Estimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

@configurable_dataclass
class EuclideanKinetic(Estimator):
    """Kinetic energy estimator in Euclidean geometry.

    Args:
        mode: Laplacian computation strategy.
        f_log_psi: Log-amplitude function (runtime dep).
        data_field: Data field containing positions (runtime dep).
    """
    mode: LaplacianMode = LaplacianMode.scan  # we use scan in this example only
    f_log_psi: WavefunctionEvaluate = runtime_dep()   # required runtime dep
    data_field: str = runtime_dep(default="electrons")  # optional runtime dep

    def evaluate_local(self, params, data, prev_local_stats, state, rngs):
        # Uses self.f_log_psi to compute the Laplacian of log|psi|
        ...
        return {"energy:kinetic": kinetic_energy}, state
```

Construct it directly — you provide the dependency yourself:

```python
kinetic = EuclideanKinetic(f_log_psi=wf.evaluate)
```

Or make it user-configurable via `cfg.get()`, then wire the runtime deps separately:

```python
kinetic = cfg.get("energy.kinetic", EuclideanKinetic())
wire(kinetic, f_log_psi=wf.evaluate)
```

The user can tune `mode` in YAML. Runtime deps like `f_log_psi` stay invisible in config:

```yaml
train:
  energy:
    kinetic:
      mode: fori_loop
```

## How the Builder Uses Wiring

When you call `VMCWorkStage.builder(cfg, wavefunction)`, it stores a context dict containing runtime objects. When you call `configure_optimizer()`, the builder resolves the optimizer from config and calls `wire()` on it:

```
optimizer = cfg.get_module("optim", ...)    # config creates the instance
wire(optimizer, f_log_psi=wf.logpsi)        # builder wires runtime deps
```

For sampling, workflows construct or resolve the sampler and pass it to `configure_sample_plan(...)`:

```
sampler = cfg.get("sampler", MCMCSampler)
builder.configure_sample_plan(wf.logpsi, {"electrons": sampler})
```

This is why the the [Wiring Principles](#wiring-principles) section says "the builder wires what it creates" — it wires components it resolves from config (such as optimizer and writers). You wire estimators yourself, either in the constructor or via `wire()`.

## Writing Custom Components

For guides on implementing each component type — what to subclass, which methods to override, and what runtime deps to expect:

```{toctree}
:hidden:

estimators
optimizers
samplers
writers
```

:::::{grid} 1 2 2 2
:gutter: 3

::::{grid-item-card} Estimators
:link: estimators
:link-type: doc

Compute physical observables. Implement `evaluate_local` for single-walker logic; the base class vmaps it over walkers.
::::

::::{grid-item-card} Optimizers
:link: optimizers
:link-type: doc

Transform gradients into parameter updates. Implement the `OptimizerLike` protocol.
::::

::::{grid-item-card} Samplers
:link: samplers
:link-type: doc

Propose and accept/reject electron moves. Implement the `SamplerLike` protocol.
::::

::::{grid-item-card} Writers
:link: writers
:link-type: doc

Record per-step statistics. Subclass `Writer` and implement `write` and `open`.
::::

:::::

For how the config system itself works (`cfg.get()`, `cfg.get_module()`, `cfg.get_collection()`), see <project:../configuration.md>.
