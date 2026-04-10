# Custom Optimizers

Most users don't need a custom optimizer — KFAC, stochastic reconfiguration, and the Optax wrappers (Adam, LAMB, etc.) cover standard use cases. Write one when you need a novel update rule that can't be expressed as an Optax transform chain, or when your optimizer needs access to the wavefunction during updates (as natural gradient methods do).

## How the Optimizer Fits In

At each training step, the training loop:

1. Computes parameter gradients (via the `LossAndGrad` estimator).
2. Calls your optimizer's `update(grads, state, params)`.
3. Adds the returned `updates` to `params`.

Your optimizer transforms raw gradients into parameter updates. For a simple optimizer like Adam, that means maintaining moving averages and scaling. For natural gradient methods, that means estimating the Fisher information matrix using `f_log_psi` — which is why the wavefunction is available as a runtime dependency.

The `init` method runs once before training starts. Use it to set up whatever your optimizer needs to carry across steps — accumulators, moving averages, preconditioning state.

## Building an Optimizer

Implement the {class}`~jaqmc.optimizer.base.OptimizerLike` protocol:

```python
from typing import Any

from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

@configurable_dataclass
class MyOptimizer:
    learning_rate: float = 1e-3       # config field — tunable via YAML
    f_log_psi: Any = runtime_dep()    # wired by builder
```

`learning_rate` is a config field — users can tune it in YAML. `f_log_psi` is a runtime dependency — the builder wires it automatically via `configure_optimizer()`. You only need `f_log_psi` if your update rule evaluates the wavefunction (natural gradient methods do; standard first-order methods don't).

**`init`** creates the optimizer state — a pytree that persists across steps:

```python
    def init(self, params, **kwargs):
        # Build state with the same tree structure as params.
        # Example: per-parameter moving averages for Adam-like methods.
        return MyState(
            step=0,
            momentum=jax.tree.map(jnp.zeros_like, params),
        )
```

`**kwargs` may include `batched_data` and `rngs` — use them if your optimizer needs the current walker positions or randomness during initialization (e.g., for initial curvature estimates).

**`update`** transforms gradients into parameter updates:

```python
    def update(self, grads, state, params, **kwargs):
        # Transform grads using your update rule.
        # `updates` must have the same pytree structure as params.
        updates = jax.tree.map(lambda g: -self.learning_rate * g, grads)
        new_state = state._replace(step=state.step + 1)
        return updates, new_state
```

The returned `updates` are added to `params` by the training loop (i.e., `new_params = params + updates`), so negate the gradients if you're doing gradient descent. `**kwargs` again may include `batched_data` and `rngs` for per-step data access.

## Getting Started

- {class}`~jaqmc.optimizer.sr.SROptimizer` — readable natural gradient implementation. Good template if your optimizer needs `f_log_psi`.
- {class}`~jaqmc.optimizer.kfac.kfac.KFACOptimizer` — more complex, but shows how to integrate an external library (kfac-jax).
- For wrapping an existing Optax optimizer, see the `optax:` module path pattern in the <project:/guide/optimizers/index.md>.

## See Also

- <project:/guide/optimizers/index.md> — background on optimizer choices and configuration
- <project:/api-reference/optimizers.md> — protocol definition and built-in optimizer API
