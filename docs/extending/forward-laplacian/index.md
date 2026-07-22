# Forward Laplacian

Computing the Laplacian of a wavefunction is often the most time-consuming task during wavefunction optimization. Since the Laplacian is a second-order operator, a naive calculation requires a forward pass and a backpropagation.
Forward Laplacian is a technique for computing the Laplacian $\nabla^2 f$ and gradient $\nabla f$ of a function $f$ with a single forward pass. Compared with the naive approach, this reduces the cost by approximately a factor of two. It also lets us better exploit sparsity in the function.

The core idea is to maintain a triple
$$
(f, \nabla f, \nabla^2 f)
$$
for every intermediate value $f$ as a function of the input variables $x$.
In addition, since many intermediate values depend on only one or two electrons, we can exploit this sparsity to further reduce the computational cost.
For details on Forward Laplacian, refer to
[R. Li et al., *A computational framework for neural network-based
variational Monte Carlo with Forward Laplacian*, Nat Mach Intell 6, 209
(2024)](https://doi.org/10.1038/s42256-024-00794-x).

If you do not care about the Forward Laplacian internals, <project:/guide/estimators/kinetic.md> is a better starting point.

Choose the branch that matches your job:

- If you need to call {func}`~jaqmc.laplacian.forward_laplacian` directly, start
  with the usage examples on this page.
- If one helper function needs a handwritten Laplacian rule, continue to
  <project:custom-rules.md>.
- If you need to debug primitive dispatch, sparse retention, or dense fallback,
  continue to <project:internals.md>. That page covers the execution engine,
  tracing behavior, dense fallback mechanics, and primitive-level extension
  points.

## Basic Example

You can use {func}`~jaqmc.laplacian.forward_laplacian` to wrap a function and obtain its Forward Laplacian variant, just as you use {func}`jax.grad` to obtain its derivative function.

```python
from jax import numpy as jnp

from jaqmc.laplacian import forward_laplacian


result = forward_laplacian(lambda x: x**2)(jnp.array(2.0))
result.x               # f(x)
result.dense_jacobian  # gradient of f(x)
result.laplacian       # Laplacian of f(x)
```

A structured result is returned, with

- `x` for the primal output
- `dense_jacobian` for the dense gradient
- `laplacian` for the propagated Laplacian

Note that you should use `result.dense_jacobian` instead of `result.jacobian`, because the latter can return a sparse Jacobian representation instead of an array.
`dense_jacobian` has shape `(n, *x.shape)`, where `n` is the total number of
tracked scalar inputs across all tracked positional arguments. The leading axis
is the derivative basis.

When batching a function that returns the full
{class}`~jaqmc.laplacian.LapTuple`, pass {func}`jax.vmap` an `out_axes` tree
built with {meth}`~jaqmc.laplacian.LapTuple.pytree_spec`. Pass axis indices as
metadata, not derivative arrays: `LapTuple(0, 1, 0)` builds a runtime value,
whereas `pytree_spec(0, 1, 0)` builds the spec JAX expects. For batch axis
`a`, pass `a` on `x` and `laplacian`. On `jacobian`, pass `a + 1` because
axis 0 is already the derivative basis—for example `pytree_spec(0, 1, 0)` when
batching along axis 0:

```python
import jax

from jaqmc.laplacian import LapTuple, forward_laplacian

batched = jax.vmap(
    forward_laplacian(fn),
    out_axes=LapTuple.pytree_spec(0, 1, 0),
)
```

If a `LapTuple` is itself the vmapped input, apply the same per-field pattern
to `in_axes`.

If you only need one field, return that field from the mapped function instead
and keep the default `vmap` output axes.

It is also fully allowed for `f` to return a PyTree. In that case, the result is also a PyTree with the same structure, and each leaf has `x`, `dense_jacobian`, and `laplacian` attributes.

## Selecting input variables

With {func}`jax.grad`, you can select input variables using `argnums=...`. Here, however, we use a different approach.
By default, `forward_laplacian(...)` keeps the derivatives and Laplacian with respect to all input variables.
But if any input array is wrapped with {func}`~jaqmc.laplacian.make_laplacian_input`, only that array is considered an input variable.

```python
from jax import numpy as jnp

from jaqmc.laplacian import forward_laplacian, make_laplacian_input


def f(a, x):
    return a * x**2


forward_laplacian(f)(jnp.array(2.0), make_laplacian_input(jnp.array(2.0)))
# LapTuple(x=Array(8., dtype=float32), jacobian=Array([8.], dtype=float32), laplacian=Array(4., dtype=float32))
```


## Weighted Summation of Laplacian

With `make_laplacian_input`, we can do more than plain `argnums=...`-style selection.
Sometimes the kinetic operator is not a plain Laplacian over all input variables.
It can instead be a weighted sum of Laplacians, such as:
$$
\sum_i a_i \frac{\partial^2}{\partial x_i^2}.
$$
This can also be computed efficiently with Forward Laplacian by choosing the Jacobian weights as $w_i^2 = a_i$.
Take the kinetic energy on spherical geometry as an example. The second-order derivative terms are:
$$
\frac{\partial^2 f}{\partial \theta^2} + \frac{1}{r^{2}\sin^2\theta}\frac{\partial^2 f}{\partial \varphi^2},
$$
Therefore, we can choose the `weights` to be $(1, \frac{1}{\sin\theta})$:
```python
weights = jnp.stack([jnp.ones_like(theta), 1 / jnp.sin(theta)], axis=-1)
result = forward_laplacian(log_psi_fn)(make_laplacian_input(electrons, weights=weights))
grad_logpsi = result.dense_jacobian.reshape(electrons.shape) / weights
```
The propagated Laplacian already reflects the weighted operator. Dividing the
dense Jacobian by `weights` only re-expresses the gradient in the original
coordinate basis.

## Sparsity

The Forward Laplacian engine in JaQMC has built-in sparsity handling.
By sparsity, we mean that the Jacobian $\nabla f$ with respect to the input variable $x$ is sparse.
In JaQMC, we explicitly support the following two types of axis-factorized sparsity:

- One-particle features: `f[..., i, ...]` only depend on input variable `x[i]`
- Two-particle features: `f[..., i, ..., j, ...]` only depend on input variable `x[i]` and `x[j]`

These patterns are exact constraints, not hints. JaQMC keeps a sparse Jacobian
only while operations preserve the one-owner or two-owner structure. Otherwise,
the transform materializes a dense Jacobian and continues with the same public
`LapTuple` contract. Fallbacks are classified by why sparsity stopped:
`not_implemented` means the sparse model could represent the operation, but that
primitive case does not have a sparse-preserving rule yet; `unrepresentable`
means the operation itself has left the current sparse model.

Sparsity handling must be enabled explicitly because it relies on the assumptions above about the function and input structure.
To do this, mark the `sparse_axis` of the input variable. In the example above, `0` means `x[i]` and `1` means `x[:, i]`.

A concrete example is the kinetic energy calculation:

```python
def evaluate_kinetic(params, data, f_log_psi):
    input_electrons = make_laplacian_input(data["electrons"], sparse_axis=0)
    result = forward_laplacian(f_log_psi)(
        params,
        data.merge({"electrons": input_electrons}),
    )
    grad_sq = jnp.sum(result.dense_jacobian**2)
    return -0.5 * (result.laplacian + grad_sq)
```

In the example above, `data["electrons"]` has shape `(n_particles, coord_dim)`, and
`make_laplacian_input(..., sparse_axis=0)` means the features in the network are sparse along the `n_particles` axis.

## Complex Wavefunctions

Real-to-complex functions are fully supported by `forward_laplacian`. Even if the implementation uses complex arithmetic internally, the transform still differentiates with respect to the tracked inputs you passed in, and therefore works without problem on non-holomorphic operations like `abs` or `conj`.

The only caveat is representation: if your Laplacian is with respect to real coordinates, keep those coordinates as the tracked inputs instead of collapsing them into complex elements.

## Next Steps

- <project:custom-rules.md> when a helper function deserves a handwritten
  `@custom_laplacian` rule
- <project:internals.md> for the interpreter pipeline and primitive-level
  extension points, including how to debug primitive dispatch, sparse
  retention, and dense fallback boundaries

```{toctree}
:maxdepth: 1
:hidden:

custom-rules.md
internals.md
```
