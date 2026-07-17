# Custom Laplacian Rules

{func}`~jaqmc.laplacian.custom_laplacian` is the Forward Laplacian analogue of `jax.custom_jvp`.
It allows you to define a custom Forward Laplacian implementation for a specific function, either for faster evaluation or for other customized logic.
If the problem really comes from a lower-level JAX primitive that affects many call sites, use
<project:internals.md> instead.

## Basic Pattern

```python
from jaqmc.laplacian import LapTuple, custom_laplacian


@custom_laplacian
def square(x):
    return x**2


@square.def_laplacian_rule
def square_rule(x):
    input_jacobian = x.dense_jacobian
    value = x.x**2
    jacobian = 2 * x.x * input_jacobian
    laplacian = 2 * x.x * x.laplacian + 2 * (input_jacobian**2).sum(axis=0)
    return LapTuple(value, jacobian, laplacian)
```

Outside `forward_laplacian`, `square` still behaves like an ordinary Python
function. Inside `forward_laplacian`, the registered rule runs whenever at
least one positional argument is being tracked.

If you decorate a function with `@custom_laplacian` but never register
`def_laplacian_rule(...)`, the function still works outside `forward_laplacian`,
but it raises an error if the transformed path reaches it.

This page focuses on the helper-level custom-rule API. The underlying staged
primitive and registry machinery stay below that public surface and usually do
not matter unless you are debugging the internals of custom-rule dispatch.

If a custom rule only handles some input patterns, it can raise
{class}`~jaqmc.laplacian.AutoLaplacianFallback` for the unsupported cases. That
delegates the current call back to the interpreter's dense automatic-rule path
instead of forcing you to encode every branch in one handwritten rule.

## More on Writing Rules

The custom rule takes the same positional arguments as the original function,
but each argument or PyTree leaf can be either a
{class}`~jaqmc.laplacian.LapTuple` or a plain array.
The return value must match the original output PyTree structure, with the
Jacobians and Laplacian propagated correctly.

If the custom handler does not support the internal sparse payloads, access the Jacobian via
{attr}`~jaqmc.laplacian.LapTuple.dense_jacobian`.
Only use {attr}`~jaqmc.laplacian.LapTuple.jacobian` when you plan to handle sparsity.
Dense Jacobians use a leading derivative basis, so rule algebra should sum over
axis `0` when contracting basis contributions. Scalar-output Jacobians may need
to broadcast over value axes; use `dense_jacobian` rather than assuming the raw
`jacobian` storage already has every output axis materialized.

When registering a rule, JaQMC warns if the rule closes over nontrivial Python
state. Because rules are recovered lazily from a registry, later mutations of
captured state may not be visible to JAX staging. Prefer passing changing
values as explicit function arguments.

Keyword arguments are not supported through the custom primitive. If you need
fixed keyword arguments, bind them first with {func}`~functools.partial` or wrap the
function before decorating it.

If all arguments are plain arrays, the transform takes the ordinary primal path
and your custom rule is not involved.
