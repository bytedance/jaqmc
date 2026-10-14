# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public annotations for selecting KFAC parameter policies."""

import jax
import kfac_jax

FIRST_ORDER_TAG = "first_order"

# JAX < 0.6 lacks a shard_map check_vma (check_rep) rule for kfac_jax's layer_tag
# primitive; register a passthrough rule so tagged functions work under shard_map.
if jax.__version_info__ < (0, 6):
    from jax.experimental import shard_map as _shard_map

    _layer_tag = kfac_jax.layers_and_loss_tags.layer_tag

    @_shard_map.register_check(_layer_tag)  # type: ignore[attr-defined]
    def _layer_tag_rep(mesh, *in_rep, meta, **_):
        [rep] = [in_rep[i] for i in meta.outputs_index]
        return rep

    _shard_map.register_norewrite(_layer_tag)  # type: ignore[attr-defined]


def register_first_order_parameter(
    output: jax.Array, parameter: jax.Array
) -> jax.Array:
    """Exclude a parameter from KFAC preconditioning.

    Block preconditioning leaves the learning-rate-scaled gradient unchanged,
    still subject to the optimizer's global norm constraint.

    Args:
        output: The output produced with ``parameter``.
        parameter: The parameter to update without KFAC preconditioning.

    Returns:
        The unchanged output with a first-order KFAC layer tag.
    """
    return kfac_jax.layers_and_loss_tags.layer_tag.bind(
        output,
        parameter,
        meta=kfac_jax.LayerMetaData(
            variant=FIRST_ORDER_TAG,
            inputs_index=(),
            outputs_index=(0,),
            params_index=(1,),
        ),
    )
