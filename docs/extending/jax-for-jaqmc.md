# JAX for JaQMC

You do **not** need to master all of JAX before extending JaQMC. You do need a working understanding of the small set of JAX ideas that appear in JaQMC's extension points, examples, and error messages.

This page explains what to learn and why it matters in JaQMC. For full tutorials, API details, and edge cases, use the links to the [official JAX documentation](inv:jax:*:doc#index).

## A Concrete JaQMC Pattern First

Before diving into terminology, here's the core extension pattern you will use in JaQMC:

```python
from jax import numpy as jnp

from collections.abc import Mapping
from typing import Any

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.array_types import Params, PRNGKey
from jaqmc.estimator import PerWalkerEstimator


class PotentialEnergy(PerWalkerEstimator):
    def evaluate_single_walker(
        self,
        params: Params,
        data: MoleculeData,
        prev_walker_stats: Mapping[str, Any],
        state: object,
        rngs: PRNGKey,
    ) -> tuple[dict[str, jnp.ndarray], object]:
        del params, prev_walker_stats, rngs
        r = jnp.linalg.norm(data["electrons"][0] - data["atoms"][0])
        return {"energy:potential": -1.0 / r}, state
```

What this example shows:

- You write logic for **one walker** in {meth}`~jaqmc.estimator.PerWalkerEstimator.evaluate_single_walker`.
- JaQMC's {class}`~jaqmc.estimator.PerWalkerEstimator` class batches that logic with {func}`jax.vmap` in {meth}`~jaqmc.estimator.PerWalkerEstimator.evaluate_batch_walkers`.
- Workflow stages may compile the batched computation with {func}`jax.jit`.

Keep this "single walker -> `vmap` -> `jit`" model in mind while reading the rest of the page.

## What to Learn (and in What Order)

Before you start extending JaQMC, make sure you are comfortable with:

- {mod}`jax.numpy` (imported as `jnp` throughout JaQMC), so you are comfortable writing array-based code
- {func}`jax.jit`, {func}`jax.vmap`, and {func}`jax.grad`, because JaQMC relies heavily on compilation, batching, and automatic differentiation
- pytrees and PRNG keys, because JaQMC passes structured state and randomness through many interfaces
- the basic Flax [`Module`](inv:flax:py:class#flax.linen.Module) / `init` / `apply` model, because JaQMC wavefunctions are Flax modules

If you are learning JAX specifically to extend JaQMC, this order is enough:

1. Start with the JaQMC mental model below: single-walker code, then `vmap`, then `jit`.
2. Learn the core JAX ideas in this page: `jax.numpy`, `jax.vmap`, `jax.jit`, and `jax.grad`.
3. Then learn pytrees, PRNG keys, and JAX-friendly control flow.
4. Learn enough Flax to understand `Module`, `init`, and `apply`.
5. Then return to JaQMC and read <project:runtime-data-conventions.md>, <project:writing-workflows.md>, and <project:wavefunctions.md>.

If you prefer learning by doing, keep this page open as a reference while you work through the extension guides.

## The JaQMC Mental Model

A **walker** is one sampled electron configuration. The most important JaQMC-specific pattern is this:

- You often write code for **one walker**.
- JaQMC batches that code over many walkers with `jax.vmap`.
- JaQMC may then compile the batched computation with `jax.jit`.

That pattern explains several design choices you will see in the docs:

- A wavefunction `__call__` usually receives one JaQMC [Data](#api-wavefunctions-data) object, not a batch.
- Estimators often inherit from {class}`~jaqmc.estimator.PerWalkerEstimator` and implement {meth}`~jaqmc.estimator.PerWalkerEstimator.evaluate_single_walker` for one walker, while `PerWalkerEstimator` handles batching.
- Runtime state and parameters must be JAX-friendly pytrees so they can pass through transforms cleanly.

If that pattern feels strange at first, focus on understanding `vmap` before worrying about more advanced JAX topics.

## What Matters Most in JaQMC

With that mental model in place, here are the JAX concepts that matter most when you extend JaQMC.

In JAX, a **transform** is a tool that takes a function and gives you back a new function with extra behavior, such as compilation, batching, or differentiation. You will see that idea repeatedly in JaQMC.

### Array-first Thinking with `jax.numpy`

`jax.numpy` is JAX's NumPy-like array library. You write computations on whole arrays, not one element at a time, which matters in JaQMC because electron positions, parameters, and statistics are all arrays and the code expects vectorized operations rather than Python loops over walkers. For more, see [Thinking in JAX](inv:jax:*:doc#notebooks/thinking_in_jax).

### `jax.jit`

`jax.jit` is a transform that takes a Python function and returns a compiled version of it. JaQMC uses it on performance-critical code, so array shapes and branching behavior need to be written in ways JAX can analyze ahead of time. For more, see [`jax.jit`](inv:jax:py:function#jax.jit).

### `jax.vmap`

`jax.vmap` is a transform that takes code written for one example and automatically applies it across a batch. This is central to JaQMC because many interfaces ask you to write logic for a **single walker**, then JaQMC applies that logic across many walkers automatically. For more, see [`jax.vmap`](inv:jax:py:function#jax.vmap).

### `jax.grad`

`jax.grad` is a transform that returns a new function that computes derivatives automatically. JaQMC relies on it because training and several estimators need derivatives of the wavefunction with respect to model parameters or electron coordinates. For more, see [`jax.grad`](inv:jax:py:function#jax.grad) and [Automatic differentiation](inv:jax:*:doc#automatic-differentiation).

### PyTrees

A pytree is JAX's name for a nested container like a tuple, dict, or dataclass whose leaves are arrays. This matters in JaQMC because parameters, optimizer state, sampler state, and `Data` objects are not always single arrays, so the codebase relies on pytree support throughout. For more, see [Pytrees](inv:jax:*:doc#pytrees).

### PRNG Keys

A PRNG key is JAX's explicit representation of random state. Instead of hidden global randomness, you pass keys through your functions, which matters in JaQMC because randomness flows through many interfaces and you usually split a key before using randomness in multiple places. For more, see [Random numbers in JAX](inv:jax:*:doc#random-numbers).

### JAX-friendly Control Flow

JAX-friendly control flow means writing `if`, loops, and branching in forms that still work when JAX compiles the function. This matters in JaQMC because inside `jit`, ordinary Python branches on array values often fail, so you usually need JAX control flow or array expressions instead. For more, see [Control flow and logical operators with JIT](inv:jax:*:doc#control-flow).

### Flax Basics

Flax basics means the small set of ideas behind Flax Linen models: `Module`, parameter initialization, and applying a model to inputs. This matters in JaQMC because wavefunctions are Flax Linen modules, so you need the basic `Module` / `init` / `apply` pattern to read and write them. For more, see [Flax basics](inv:flax:*:doc#guides/flax_fundamentals/flax_basics).

## Common JAX Pitfalls When Extending JaQMC

These are worth learning early because they appear often in real extension work:

- **Python control flow on arrays**: `if x > 0:` is often wrong when `x` is a JAX array inside compiled code. Prefer `jnp.where` or `jax.lax.cond`.
- **Dynamic shapes inside `jit`**: array shapes that depend on runtime values, for example creating an array whose length depends on the current input data, often break compilation or lead to recompilation.
- **Treating batched and unbatched data the same way**: many JaQMC hooks are intentionally single-walker APIs.
- **Reusing PRNG keys**: split keys before passing them to separate calls.
- **Putting shape-defining values in runtime data**: if a value determines shapes or control flow, it often belongs in config rather than in mutable runtime data.

Use JaQMC's docs to understand where these issues show up in this codebase. Use the [official JAX docs](inv:jax:*:doc#index) when you need the exact semantics or edge cases.

## What You Can Ignore For Now

Most JaQMC users do **not** need to learn these topics before becoming productive:

- writing custom JAX primitives
- low-level XLA details
- advanced sharding internals
- transformation implementation details

You may eventually need some of them for performance work or distributed systems research, but they are not the right starting point for learning JaQMC.

## Where to Go Next

- To **run simulations**, continue with <project:../getting-started/quick-start.md> and the system guides.
- To **understand the training loop**, read <project:../getting-started/concepts.md>.
- To **build custom components**, continue with <project:runtime-data-conventions.md>, <project:writing-workflows.md>, and <project:wavefunctions.md>.
- For JAX-specific install/runtime issues or API details, use the [official JAX documentation](inv:jax:*:doc#index).
