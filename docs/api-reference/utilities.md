# Utilities

Helper functions for array manipulation, function transformations, clipping, units, supercell construction, checkpointing, and multi-device parallelism.

## Function transforms

Wrappers for extracting real/imaginary parts and handling complex-valued JAX functions.

```{eval-rst}
.. autotype:: jaqmc.utils.func_transform.CompatibleFunc

.. autofunction:: jaqmc.utils.func_transform.with_real
.. autofunction:: jaqmc.utils.func_transform.with_imag
.. autofunction:: jaqmc.utils.func_transform.with_output
.. autofunction:: jaqmc.utils.func_transform.transform_maybe_complex
.. autofunction:: jaqmc.utils.func_transform.linearize_maybe_complex
.. autofunction:: jaqmc.utils.func_transform.grad_maybe_complex
.. autofunction:: jaqmc.utils.func_transform.hessian_maybe_complex
.. autofunction:: jaqmc.utils.func_transform.transform_with_data
```

## Array utilities

```{eval-rst}
.. autofunction:: jaqmc.utils.array.array_partitions
.. autofunction:: jaqmc.utils.array.split_nonempty_channels
.. autofunction:: jaqmc.utils.array.match_first_axis_of
```

## Clipping

```{eval-rst}
.. autofunction:: jaqmc.utils.clip.iqr_clip
.. autofunction:: jaqmc.utils.clip.iqr_clip_real
```

## Units

```{eval-rst}
.. autoclass:: jaqmc.utils.units.LengthUnit
```

## Supercell construction

```{eval-rst}
.. autofunction:: jaqmc.utils.supercell.get_reciprocal_vectors
.. autofunction:: jaqmc.utils.supercell.get_supercell_kpts
.. autofunction:: jaqmc.utils.supercell.get_supercell_copies
```

## Checkpointing

```{eval-rst}
.. autoclass:: jaqmc.utils.checkpoint.NumPyCheckpointManager
   :members:

.. autotype:: jaqmc.utils.checkpoint.PathLike

.. autofunction:: jaqmc.utils.checkpoint.tree_to_npz
.. autofunction:: jaqmc.utils.checkpoint.tree_from_npz
```

## Configuration helpers

```{eval-rst}
.. autoclass:: jaqmc.utils.config.ConfigManagerLike
   :members:
```

## Multi-device parallelism

```{eval-rst}
.. autoclass:: jaqmc.utils.parallel_jax.DistributedConfig

.. autofunction:: jaqmc.utils.parallel_jax.make_mesh
.. autofunction:: jaqmc.utils.parallel_jax.make_sharding
.. autofunction:: jaqmc.utils.parallel_jax.jit_sharded
.. autofunction:: jaqmc.utils.parallel_jax.pvary
.. autofunction:: jaqmc.utils.parallel_jax.pmean
.. autofunction:: jaqmc.utils.parallel_jax.all_gather
.. autofunction:: jaqmc.utils.parallel_jax.addressable_data
```

## Array type aliases

```{eval-rst}
.. autotype:: jaqmc.array_types.PRNGKey
.. autotype:: jaqmc.array_types.Params
.. autotype:: jaqmc.array_types.PyTree
.. autotype:: jaqmc.array_types.ArrayTree
.. autotype:: jaqmc.array_types.ArrayLikeTree
```
