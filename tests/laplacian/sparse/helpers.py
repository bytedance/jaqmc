# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Shared sparse topology fixtures and retention helpers."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.laplacian import (
    LapTuple,
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    OwnerRoles,
    make_laplacian_input,
)


def assert_retains_sparse_family(
    out: LapTuple,
    expected_family: type[Local1Jacobian] | type[Local2Jacobian],
) -> None:
    """Assert ``out`` stayed in the requested sparse Jacobian family."""
    assert isinstance(out, LapTuple)
    assert isinstance(out.jacobian, expected_family)


def repeated_owner_ids_local1_seed() -> LapTuple:
    """Local1 state with an axis-varying owner role that repeats owner ids.

    Topology:
    - output shape ``(2, 2)`` on output axis 0
    - owner role varies along output axis 0 with ids ``[2, 2]`` (repeated, not
      constant across the axis)
    - ``input_shape=(3, 2)`` with ``input_owner_axis=0`` (tracked inputs live
      along the input-owner axis, distinct from the output axis above)
    """
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    return LapTuple(
        x,
        Local1Jacobian(
            blocks=jnp.array(
                [
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                    ]
                ],
                dtype=jnp.float32,
            ),
            owners=OwnerRoles(OwnerRole(0, np.array([2, 2], dtype=np.int32))),
            input_shape=(3, 2),
            input_owner_axis=0,
        ),
        jnp.zeros_like(x),
    )


def broadcast_local1_seed() -> LapTuple:
    """Local1 state with broadcast-filled blocks and owners on axis 1.

    Topology:
    - primal/output shape ``(4, 2, 3)``
    - owner role varies along axis 1 with ids ``[0, 1]``
    - ``input_shape=(2, 3)`` with ``input_owner_axis=0``
    """
    x = jnp.arange(1.0, 25.0, dtype=jnp.float32).reshape(4, 2, 3)
    return LapTuple(
        x,
        Local1Jacobian(
            blocks=jnp.broadcast_to(
                jnp.arange(18.0, dtype=jnp.float32).reshape(1, 3, 1, 2, 3),
                (1, 3, *x.shape),
            ),
            owners=OwnerRoles(OwnerRole(1, np.array([0, 1], dtype=np.int32))),
            input_shape=(2, 3),
            input_owner_axis=0,
        ),
        jnp.zeros_like(x),
    )


def mismatched_local1_pair() -> tuple[LapTuple, LapTuple]:
    """Two Local1 operands with swapped owner ids along axis 0.

    Used to exercise Local1-to-Local2 promotion in binary arithmetic.
    """
    x = jnp.arange(4.0, dtype=jnp.float32).reshape(2, 2)
    lhs = LapTuple(
        x,
        Local1Jacobian(
            blocks=jnp.array(
                [
                    [
                        [[1.0, 3.0], [2.0, 4.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                    ]
                ],
                dtype=jnp.float32,
            ),
            owners=OwnerRoles(OwnerRole(0, np.array([0, 1], dtype=np.int32))),
            input_shape=(2, 2),
            input_owner_axis=0,
        ),
        jnp.zeros_like(x),
    )
    rhs = LapTuple(
        x,
        Local1Jacobian(
            blocks=jnp.array(
                [
                    [
                        [[5.0, 7.0], [6.0, 8.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                    ]
                ],
                dtype=jnp.float32,
            ),
            owners=OwnerRoles(OwnerRole(0, np.array([1, 0], dtype=np.int32))),
            input_shape=(2, 2),
            input_owner_axis=0,
        ),
        jnp.zeros_like(x),
    )
    return lhs, rhs


def repeated_owner_local2_seed() -> LapTuple:
    """Local2 state with a repeated owner id along output axis 0.

    Topology:
    - output shape ``(2, 1)``
    - first owner role is constant owner id 0
    - second owner role varies along output axis 0 with ids ``[0, 1]``
    - ``input_shape=(2, 1)`` with ``input_owner_axis=0``
    """
    x = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
    blocks = jnp.array(
        [
            [[[-1.0], [2.0]]],
            [[[3.0], [5.0]]],
        ],
        dtype=jnp.float32,
    )
    return LapTuple(
        x,
        Local2Jacobian(
            blocks=blocks,
            owners=OwnerRoles(
                OwnerRole(None, np.array([0], dtype=np.int32)),
                OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            ),
            input_shape=(2, 1),
            input_owner_axis=0,
        ),
        jnp.zeros_like(x),
    )


def select_n_broadcast_mixed_branches_scenario() -> tuple[
    Callable[[jnp.ndarray], jnp.ndarray],
    LapTuple,
]:
    """Return the mixed ``select_n`` graph and its production Local1 seed.

    The selector includes case ``2`` so the broadcast plain branch is selected.
    Plain operands are broadcast to the tracked output shape, exercising the
    zero-local sparse path's output-shape and owner-axis adaptation.
    """

    def select_broadcast_mixed(value: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.select_n(
            jnp.array(
                [[0, 1, 2], [1, 0, 2], [2, 1, 0]],
                dtype=jnp.int32,
            ),
            value,
            jnp.broadcast_to(value[:, :1] + 1.0, value.shape),
            jnp.broadcast_to(
                jnp.full((1, 1), 7.0, dtype=value.dtype),
                value.shape,
            ),
        )

    x = jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3)
    return select_broadcast_mixed, make_laplacian_input(x, sparse_axis=0)


def two_local1_query_key_dot_scenario() -> tuple[
    Callable[[jnp.ndarray], jnp.ndarray],
    LapTuple,
]:
    """Return the query/key dot graph and its production Local1 seed."""
    weight_q = jnp.arange(36.0, dtype=jnp.float32).reshape(6, 6) / 20.0
    weight_k = jnp.arange(36.0, dtype=jnp.float32).reshape(6, 6)[::-1] / 18.0

    def query_key_dot(value):
        return jax.lax.dot_general(
            jax.lax.reshape(
                jax.lax.dot_general(
                    value,
                    weight_q,
                    dimension_numbers=(((1,), (0,)), ((), ())),
                ),
                new_sizes=(4, 2, 3),
                dimensions=None,
            ),
            jax.lax.reshape(
                jax.lax.dot_general(
                    value,
                    weight_k,
                    dimension_numbers=(((1,), (0,)), ((), ())),
                ),
                new_sizes=(4, 2, 3),
                dimensions=None,
            ),
            dimension_numbers=(((2,), (2,)), ((1,), (1,))),
        )

    x = jnp.arange(24.0, dtype=jnp.float32).reshape(4, 6) / 10.0
    return query_key_dot, make_laplacian_input(x, sparse_axis=0)
