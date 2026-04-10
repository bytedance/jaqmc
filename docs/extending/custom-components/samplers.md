# Custom Samplers

Write a custom sampler when you need a different sampling algorithm — for example, Hamiltonian Monte Carlo (HMC), Langevin dynamics, or cluster moves. The built-in {class}`~jaqmc.sampler.mcmc.MCMCSampler` implements standard Metropolis-Hastings with adaptive step width, which works well for most systems.

## How Sampling Works

At each training step, the training loop calls your sampler's `step()` to generate new electron configurations. The sampler doesn't evaluate the wavefunction directly — instead, it receives `batch_log_prob`, a function that returns log|ψ|² for any set of positions. Your job is to:

1. **Propose** new positions (however your algorithm does it).
2. **Evaluate** `batch_log_prob` at the proposed positions.
3. **Accept or reject** based on your criterion (Metropolis-Hastings, HMC Hamiltonian, etc.).
4. **Return** the new positions, an acceptance rate, and updated state.

The training loop calls `init()` once before training starts to create the
sampler state, then calls `step()` every iteration.

Most sampler implementations never manipulate the full
{class}`~jaqmc.data.BatchedData` wrapper. Their `init()` and `step()` methods
usually receive only the batched subset for the fields they own, typically
`electrons`. If you need to understand how that subset relates to the full
runtime data object, or you are writing lower-level batch plumbing around the
sampler interface, see the [advanced batching section](#ext-runtime-data-advanced)
in <project:/extending/runtime-data-conventions.md>.

## Building a Sampler

Implement the {class}`~jaqmc.sampler.base.SamplerLike` protocol:

```python
from typing import Any

from jax import numpy as jnp

from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
```

Config fields go in YAML; runtime deps are wired by the builder:

```python
@configurable_dataclass
class HMCSampler:
    n_leapfrog: int = 10              # config field
    step_size: float = 0.1            # config field
    sampling_proposal: Any = runtime_dep(default=gaussian_proposal)
```

**`init`** creates the sampler state. This state carries adaptive parameters, counters, or anything else that persists across steps. It must be a JAX pytree (arrays, NamedTuples, dataclasses) so it can be checkpointed:

```python
    def init(self, data, rngs):
        return HMCState(step_size=jnp.array(self.step_size))
```

**`step`** is where the sampling algorithm lives. It receives `batch_log_prob` — a function that maps a batched data pytree to a 1D array of log|ψ|² values (one per walker). Note: this is log-*probability* (`2 * log|psi|`), not log-amplitude.

```python
    def step(self, batch_log_prob, data, state, rngs):
        # 1. Propose new positions
        #    For HMC: generate momentum, run leapfrog integration
        #    For MH: add noise via self.sampling_proposal

        # 2. Evaluate log|psi|^2 at proposed positions
        log_prob_proposed = batch_log_prob(proposed_data)

        # 3. Accept/reject
        #    Compare log_prob_proposed vs log_prob_current
        #    For HMC: include kinetic energy in the Hamiltonian

        # 4. Return
        return new_data, {"pmove": acceptance_rate}, new_state
```

Include `"pmove"` in the stats dict — the console writer displays it, and it's the primary diagnostic for sampling quality. A healthy acceptance rate is typically 0.3–0.7 for Metropolis-Hastings; other algorithms have different targets.

## Sampling Proposals

The `sampling_proposal` field is an optional runtime dependency — a function with signature `(rngs, x, stddev) -> x_new` that proposes new positions. It's separate from the sampler so that the same algorithm can work across different geometries:

- **Euclidean** (molecules): Gaussian noise (the default).
- **Periodic** (solids): Gaussian noise wrapped back into the simulation cell.
- **Spherical** (Hall): Moves on the sphere surface.

The workflow configures the appropriate proposal when constructing or resolving the sampler (e.g., `MCMCSampler(sampling_proposal=sphere_proposal)`), and passes the sampler to `configure_sample_plan(f_log_amplitude, {"electrons": sampler})`. If your algorithm generates proposals internally (as HMC does via leapfrog integration), you don't need this field.

## Getting Started

{class}`~jaqmc.sampler.mcmc.MCMCSampler` is a complete reference implementation. It shows Metropolis-Hastings accept/reject, adaptive step-width tuning (adjusting `stddev` to keep `pmove` in a target range), and how state flows through the step cycle.

## See Also

- <project:/guide/sampling.md> — background on MCMC tuning and diagnostics
- <project:/api-reference/samplers.md> — protocol definition and MCMCSampler API
