# Writing Wavefunctions

The <project:writing-workflows.md> tutorial showed how a wavefunction fits into a workflow — subclass {class}`~jaqmc.wavefunction.Wavefunction`, implement `__call__`, and pass `wf.evaluate` to the builder. This page clarifies the `__call__`/`apply`/`evaluate` interfaces, then covers the reusable building blocks for complex architectures and how to make a wavefunction YAML-configurable. Both levels build on [Flax basics](inv:flax:*:doc#guides/flax_fundamentals/flax_basics). If terms like `jit`, `vmap`, pytrees, or `Module.apply` still feel unfamiliar, read <project:../extending/jax-for-jaqmc.md> first.

## The `__call__` Contract

{class}`~jaqmc.wavefunction.Wavefunction` is a Flax [`nn.Module`](inv:flax:py:class#flax.linen.Module) with one abstract method:

```python
class Wavefunction[DataT: Data, OutputT](nn.Module, ABC):
    @abstractmethod
    def __call__(self, data: DataT) -> OutputT: ...
```

The type parameters are:

- **`DataT`** — your data type, a subclass of {class}`~jaqmc.data.Data` (e.g., `HydrogenAtomData`, `MoleculeData`). `Data` is a JAX-compatible dataclass that flows through `jit`, `grad`, and `vmap`.
- **`OutputT`** — the return type. A scalar `jnp.ndarray` for simple wavefunctions, or a `TypedDict` like {class}`~jaqmc.wavefunction.output.logdet.RealLogDetOutput` for architectures that return additional information (sign, per-determinant values).

Both are optional — `class MyWF(Wavefunction):` works fine when you don't need the type constraints, as the hydrogen atom example shows.

The base class provides two methods for free:

- **`init_params(data, rngs)`** — initializes parameters by calling Flax's `self.init(rngs, data)`.
- **`evaluate(params, data)`** — runs the forward pass by calling `self.apply(params, data)`.

These are what the rest of JaQMC (samplers, optimizers, estimators) interact with. You only implement `__call__`.

`__call__` receives a single walker — one `Data` instance, not a batch. JaQMC handles batching externally with `jax.vmap`, so your implementation never needs to think about the batch dimension. It also runs inside `jax.jit`, so avoid Python-level data-dependent control flow — use `jax.lax.cond` or `jnp.where` instead of `if` statements that branch on array values. <project:../extending/jax-for-jaqmc.md> explains why this single-walker-plus-`vmap` pattern appears throughout the framework.

For built-in-style wavefunctions, treat `data.electrons` as one walker's
particle coordinates, typically with shape `(n_particles, ndim)`. You only need
to think about `BatchedData` once you start writing lower-level sampler,
workflow, or estimator plumbing that manipulates whole walker batches directly;
see <project:runtime-data-conventions.md>. Even if
`data_init` or sampling code is working with batch-shaped arrays elsewhere, the
wavefunction contract here stays single-walker.

## Execution Interfaces: `__call__` vs `apply` vs `evaluate`

These three names are related but serve different layers:

| Interface | Signature | Owned by | Typical caller |
|--------|---------|---------|---------|
| `__call__` | `(data) -> OutputT` | Your wavefunction subclass | Flax internals via `apply` |
| `apply` | `(variables, *args, method=...) -> Any` | Flax `nn.Module` | Advanced users and internal wrappers |
| `evaluate` | `(params, data) -> OutputT` | JaQMC `Wavefunction` base class | Workflow, sampler, optimizer, estimator wiring |

- Use **`__call__`** to define model math for one walker.
- Use **`evaluate`** as the default JaQMC-facing callable with explicit parameters.
- Use **`apply`** directly only for advanced cases (for example, calling an alternate method like `get_orbitals`).

Framework call path in practice:

```text
workflow/estimator/sampler -> wf.logpsi or wf.evaluate
                          -> wf.evaluate(params, data)
                          -> wf.apply(params, data)
                          -> wf.__call__(data)
```

Minimal example:

```python
class MyWF(Wavefunction[MyData, jnp.ndarray]):
    @nn.compact
    def __call__(self, data: MyData) -> jnp.ndarray:
        alpha = self.param("alpha", lambda *_: jnp.array(0.0))
        return alpha * jnp.linalg.norm(data.electrons)

wf = MyWF()
params = wf.init_params(data, rngs)
value = wf.evaluate(params, data)           # framework contract
value2 = wf.apply(params, data)             # equivalent low-level Flax call
```

### Structured Return Types

Production wavefunctions (FermiNet, Psiformer) return more than a scalar — they also provide the sign of the wavefunction, which is needed for energy calculations involving pseudopotentials. They return {class}`~jaqmc.wavefunction.output.logdet.RealLogDetOutput`:

```python
from jaqmc.wavefunction.output.logdet import RealLogDetOutput

class MyWavefunction(Wavefunction[MoleculeData, RealLogDetOutput]):
    def __call__(self, data: MoleculeData) -> RealLogDetOutput:
        ...
        return RealLogDetOutput(
            logpsi=log_amplitude,       # log|psi| (scalar)
            sign_logpsi=sign,           # sign of psi (+1 or -1)
            sign_logdets=signs,         # signs of individual determinants
            abs_logdets=logdets,        # log|det| for each determinant
        )
```

If your `__call__` returns `RealLogDetOutput`, the [extraction methods](#extraction-methods) below become one-liner delegations to `evaluate`.

## Reusable Building Blocks

The `jaqmc.wavefunction` package provides Flax modules for the common stages of a molecular wavefunction. You can compose them to build new architectures while only implementing the novel part — typically the backbone.

The built-in wavefunctions (FermiNet, Psiformer) follow this pattern:

1. **Input features** — construct atom-electron and electron-electron feature vectors from raw positions.
2. **Backbone** — transform those features through interaction layers (message-passing in FermiNet, self-attention in Psiformer) to produce per-electron representations.
3. **Orbital projection** — project the per-electron representations into orbital matrices, one per determinant.
4. **Envelope** — multiply each orbital by a distance-dependent envelope that enforces the correct asymptotic decay.
5. **Log-determinant** — compute the log-sum of Slater determinants to produce the final log|ψ| and sign.

This isn't a required architecture — the only hard contract is `__call__`. A wavefunction that skips the orbital/determinant machinery entirely (like the hydrogen atom example) is perfectly valid. But when you *do* want determinant-based antisymmetry, these modules save you from reimplementing the standard stages.

See the <project:../api-reference/wavefunctions.md> for the full list of available modules and their parameters.

### Example: Custom Backbone with Standard I/O

The most common extension point is the backbone — the layers that transform input features into single-electron representations. Here's how to write a custom backbone while reusing the standard input, orbital projection, envelope, and log-determinant layers:

```python
from jaqmc.app.molecule.data import MoleculeData
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction import Wavefunction
from jaqmc.wavefunction.input.atomic import MoleculeFeatures
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType
from jaqmc.wavefunction.output.logdet import LogDet, RealLogDetOutput
from jaqmc.wavefunction.output.orbital import OrbitalProjection


class MyBackbone(nn.Module):
    """Your custom interaction layers."""
    nspins: tuple[int, int]
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, h_one, h_two):
        # h_one: (n_electrons, feature_dim) — single-electron features
        # h_two: (n_electrons, n_electrons, feature_dim) — pairwise features
        ...
        return h_one  # per-electron representations for orbital projection


class MyWavefunction(Wavefunction[MoleculeData, RealLogDetOutput]):
    nspins: tuple[int, int] = runtime_dep()  # set by workflow; see below
    ndets: int = 8
    hidden_dim: int = 128

    def setup(self):
        self.features = MoleculeFeatures()
        self.backbone = MyBackbone(self.nspins, self.hidden_dim)
        self.orbitals = OrbitalProjection(nspins=self.nspins, ndets=self.ndets)
        self.envelope = Envelope(envelope_type=EnvelopeType.abs_isotropic,
                                 ndets=self.ndets, nspins=self.nspins)
        self.logdet = LogDet()

    def __call__(self, data: MoleculeData):
        emb = self.features(data.electrons, data.atoms)
        h_one = self.backbone(emb["ae_features"], emb["ee_features"])
        orbs = self.orbitals(h_one) * self.envelope(emb["ae_vec"], emb["r_ae"])
        return self.logdet(orbs)
```

`runtime_dep()` marks fields whose values come from the workflow rather than from user config — `nspins` is determined by the molecular system, so the workflow sets it at startup. Accessing a `runtime_dep()` field before the workflow sets it raises `AttributeError` with a descriptive message. See [Making it YAML-configurable](#making-it-yaml-configurable) for more.

## Extraction Methods

For the hydrogen atom, `evaluate` is all you need — it returns a scalar and every consumer uses it directly. Production wavefunctions return richer output (like {class}`~jaqmc.wavefunction.output.logdet.RealLogDetOutput`), and different consumers need different slices. Extraction methods give each consumer exactly the interface it needs:

| Method | Returns | Used by |
|--------|---------|---------|
| `logpsi(params, data)` | $\log\lvert\psi\rvert$ (scalar) | VMC loss, MCMC sampling |
| `phase_logpsi(params, data)` | $(\operatorname{sgn}\psi,\;\log\lvert\psi\rvert)$ | Pseudopotential estimator (needs sign for wavefunction ratios) |

When `__call__` returns {class}`~jaqmc.wavefunction.output.logdet.RealLogDetOutput`, both are one-liner extractions from `evaluate`:

```python
def logpsi(self, params, data):
    return self.evaluate(params, data)["logpsi"]

def phase_logpsi(self, params, data):
    out = self.evaluate(params, data)
    return out["sign_logpsi"], out["logpsi"]
```

:::{admonition} Molecule wavefunction protocol
:class: note

The molecule and solid apps define a protocol that formalizes which extraction
methods they expect. The workflow validates it at startup — if you forget a
method, you'll get a clear error. The protocol also includes an `orbitals`
method for pretraining against Hartree-Fock references — see
{ghsrc}`src/jaqmc/app/molecule/wavefunction/ferminet.py` for the implementation
pattern.
:::

### Making It YAML-Configurable

To let users select your wavefunction from the CLI, the class must be importable via the [module path syntax](#swappable-modules). Fields fall into two categories:

- **{func}`~jaqmc.utils.wiring.runtime_dep`** — set by the workflow from the system config (e.g., `nspins` from the molecular geometry). Not user-configurable.
- **Regular fields with defaults** (e.g., `ndets: int = 8`) — configurable via YAML or CLI overrides.

Unlike estimators/optimizers/samplers, wavefunction classes do **not** need
{deco}`~jaqmc.utils.config.configurable_dataclass`: the {class}`~jaqmc.wavefunction.Wavefunction` base class automatically handles
configuration serialization for subclasses and excludes Flax internal fields
(`parent`, `name`). For non-wavefunction components, see
<project:custom-components/index.md>, which uses
{deco}`~jaqmc.utils.config.configurable_dataclass`,
{func}`~jaqmc.utils.wiring.runtime_dep`, and
{func}`~jaqmc.utils.wiring.wire`.

On the workflow side, {meth}`~jaqmc.utils.config.ConfigManager.get_module`
resolves the class from config and instantiates it with the user's field
overrides. The workflow then sets runtime dependencies before passing the
wavefunction to the builder:

```python
wf = cfg.get_module("wf", "jaqmc.app.molecule.wavefunction.ferminet")
wf.nspins = system_config.electron_spins  # set runtime_dep before use

train = VMCWorkStage.builder(cfg.scoped("train"), wf)
sampler = cfg.get("sampler", MCMCSampler)
train.configure_sample_plan(wf.logpsi, {"electrons": sampler})
# ... rest of wiring as in the workflows tutorial
```

Run with:

```bash
jaqmc molecule train wf.module=my_package.my_wf:MyWavefunction wf.ndets=16 wf.hidden_dim=256
```

## Where to Look

- {ghsrc}`src/jaqmc/app/molecule/wavefunction/ferminet.py` — Complete FermiNet implementation (best template to copy).
- {ghsrc}`src/jaqmc/wavefunction/base.py` — `Wavefunction` base class and protocol definitions.
- {ghsrc}`src/jaqmc/app/molecule/wavefunction/base.py` — molecule wavefunction protocol definition.
- {ghsrc}`src/jaqmc/app/molecule/wavefunction/psiformer.py` — Psiformer implementation.
- {ghsrc}`src/jaqmc/app/molecule/workflow.py` — How the workflow resolves and wires the wavefunction.
