# Forward Laplacian

Reference for the public names in {mod}`jaqmc.laplacian`.

If you are deciding how to use the transform, what contracts it enforces, or
which extension path you need, start with
<project:/extending/forward-laplacian/index.md>.

Most callers only need {func}`~jaqmc.laplacian.forward_laplacian`,
{func}`~jaqmc.laplacian.make_laplacian_input`, and
{class}`~jaqmc.laplacian.LapTuple`. If you only consume transform outputs, read
derivatives through {attr}`~jaqmc.laplacian.LapTuple.dense_jacobian` and ignore
the sparse Jacobian storage types below. Those types matter when you inspect
storage details or write extensions. For the internal sparse propagation model
and dense-fallback boundaries, see
<project:/extending/forward-laplacian/internals.md>.

## Core transform API

These are the normal entrypoints for calling the transform and consuming its
result.

- {func}`~jaqmc.laplacian.forward_laplacian` wraps a function and returns a
  Forward Laplacian version that propagates value, Jacobian, and Laplacian
  together.
- {func}`~jaqmc.laplacian.make_laplacian_input` seeds one array or pytree for
  tracking. Use it when you want to track only specific inputs or when you want
  weighted or sparse seeding instead of the default dense identity seed.
- {class}`~jaqmc.laplacian.LapTuple` is the result container returned by the
  transform. It carries the primal output, Jacobian payload, and propagated
  Laplacian.

```{eval-rst}
.. autofunction:: jaqmc.laplacian.forward_laplacian

.. autofunction:: jaqmc.laplacian.make_laplacian_input

.. autoclass:: jaqmc.laplacian.LapTuple
   :members:
   :undoc-members:
```

## Custom rule API

These APIs are for specializing one helper function without changing the
primitive registry.

- {func}`~jaqmc.laplacian.custom_laplacian` decorates a function and adds a
  `.def_laplacian_rule(...)` registration hook. The handwritten rule is
  registered separately through that hook.
- {class}`~jaqmc.laplacian.AutoLaplacianFallback` can be raised from a
  registered custom rule to delegate unsupported cases back to the ordinary
  automatic path.

Use this API when you want a local function-level override. Use
{mod}`jaqmc.laplacian.primitives` only when you need to extend the lower-level
primitive registry itself.

```{eval-rst}
.. autofunction:: jaqmc.laplacian.custom_laplacian

.. autoexception:: jaqmc.laplacian.AutoLaplacianFallback
```

## Core types

{class}`~jaqmc.laplacian.LapTuple` is the central runtime value exposed by the
transform. It carries:

- `x`: the primal output
- `jacobian`: the Jacobian payload in either dense or structured sparse form
- `laplacian`: the propagated Laplacian with the same shape as `x`

When you batch or shard a function that returns a
{class}`~jaqmc.laplacian.LapTuple` with {func}`jax.vmap` or `shard_map`, pass
axis or sharding metadata through {meth}`~jaqmc.laplacian.LapTuple.pytree_spec`
rather than constructing one with derivative arrays. See
<project:/extending/forward-laplacian/index.md> for the `vmap` axis
convention; use the same per-field pattern with `PartitionSpec` leaves for
`shard_map`.

{obj}`~jaqmc.laplacian.ArrayOrLapTuple` is the common union type used by the
public transform and low-level handlers for values that may or may not already
be carrying derivative state.

```{eval-rst}
.. autotype:: jaqmc.laplacian.ArrayOrLapTuple
```

## Guard helpers

The guard helpers answer two common questions:

- "Is this value carrying Forward Laplacian state at all?"
- "If so, what Jacobian representation is it carrying?"

Use {func}`~jaqmc.laplacian.is_laptuple` to detect tracked values in general.
Use the more specific guards when behavior depends on whether the Jacobian is
dense, `Local1`, `Local2`, or sparse in general.

```{eval-rst}
.. autofunction:: jaqmc.laplacian.is_laptuple

.. autofunction:: jaqmc.laplacian.is_dense_laptuple

.. autofunction:: jaqmc.laplacian.is_local1_laptuple

.. autofunction:: jaqmc.laplacian.is_local2_laptuple

.. autofunction:: jaqmc.laplacian.is_sparse_laptuple

.. autofunction:: jaqmc.laplacian.is_sparse_jacobian
```

## Sparse Jacobian model

The sparse Jacobian types describe the structured owner-local storage used when
the transform can preserve sparsity exactly.

Sparse block payloads always use the full output layout:
`blocks.shape[:-2] == output_shape == LapTuple.x.shape`. Broadcasting therefore
uses full blocks produced by `jnp.broadcast_to`; the sparse API has no separate
compressed or logical output-shape metadata.

- {class}`~jaqmc.laplacian.Local1Jacobian` represents outputs that depend on
  one owner entry of the chosen input axis.
- {class}`~jaqmc.laplacian.Local2Jacobian` represents outputs that depend on at
  most two owner entries of that axis.
- {class}`~jaqmc.laplacian.OwnerRole` and
  {class}`~jaqmc.laplacian.OwnerRoles` describe which owner ids a sparse slot
  refers to.
- {class}`~jaqmc.laplacian.SparseJacobian` is the shared public type for these
  sparse payload families.

```{eval-rst}
.. automodule:: jaqmc.laplacian.sparse
   :no-members:
   :no-index:

.. autoclass:: jaqmc.laplacian.OwnerRole
   :members:

.. autoclass:: jaqmc.laplacian.OwnerRoles
   :members:

.. autoclass:: jaqmc.laplacian.Local1Jacobian
   :members:
   :inherited-members:

.. autoclass:: jaqmc.laplacian.Local2Jacobian
   :members:
   :inherited-members:

.. autotype:: jaqmc.laplacian.SparseJacobian
```

## Primitive extension helpers

Primitive registry and wrapper-construction helpers are documented under the
related submodule {mod}`jaqmc.laplacian.primitives`, not under the top-level
{mod}`jaqmc.laplacian` export surface.

Use that submodule when you need to register or inspect primitive handlers with
`register_function`, `deregister_function`, or `get_laplacian`, or when you are
authoring handlers with `setup_handler` and the `wrap_*` helper families.

For the extension workflow and internal helper utilities used when authoring
custom handlers, see <project:/extending/forward-laplacian/internals.md>.
