# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Forward Laplacian input fixtures for tests."""

from math import prod
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.laplacian import LapTuple
from jaqmc.laplacian.sparse import Local1Jacobian, Local2Jacobian, OwnerRole, OwnerRoles

type InputCase = Literal["plain", "dense", "local1", "local2"]
type RealInputDomain = Literal["real", "positive", "unit"]
type InputDomain = Literal["real", "positive", "unit", "complex"]

# The interval a named domain keeps inputs inside: "positive" stays well clear
# of the singularity at 0 (log, sqrt, division), "unit" stays inside
# arcsin/arccos's [-1, 1]. None means unrestricted.
DOMAIN_BOUNDS: dict[RealInputDomain, tuple[float, float] | None] = {
    "real": None,
    "positive": (0.3, 2.0),
    "unit": (-0.8, 0.8),
}


def random_array(
    domain: InputDomain = "real",
    shape=(3, 3),
    key: int = 0,
) -> jnp.ndarray:
    """Draw a deterministic array of tracked coordinates inside ``domain``."""
    if domain == "complex":
        return random_array("real", shape, key) + 1j * random_array(
            "real", shape, key + 1
        )
    rng = jax.random.PRNGKey(1234 + key)
    bounds = DOMAIN_BOUNDS[domain]
    if bounds is None:
        return jax.random.normal(rng, shape)
    return jax.random.uniform(rng, shape, minval=bounds[0], maxval=bounds[1])


# Complex inputs: tracked coordinates are always real. A complex value is
# packed as a trailing length-2 (real, imag) axis of x and assembled *inside*
# the traced function via to_complex, so Jacobian and Laplacian stay in the
# real coordinate basis everything above assumes.

# Packed base coordinates: 3 owners, one complex scalar (2 reals) each.
VECTOR_SHAPE = (3, 3, 2)

# Packed complex 2x2 matrices: 2 owners of one matrix row (2 complex) each.
MATRIX_SHAPE = (2, 2, 2)


def to_complex(packed: jnp.ndarray) -> jnp.ndarray:
    """View a packed ``(..., 2)`` real array as a complex array."""
    return packed[..., 0] + 1j * packed[..., 1]


def make_local1_input(
    x: jnp.ndarray,
    *,
    blocks: jnp.ndarray,
    owners: OwnerRoles,
    input_shape: tuple[int, ...],
    input_owner_axis: int = 0,
    laplacian: jnp.ndarray | None = None,
) -> LapTuple:
    """Build an explicit Local1 Forward Laplacian test input."""
    return LapTuple(
        x,
        Local1Jacobian(
            blocks=blocks,
            owners=owners,
            input_shape=input_shape,
            input_owner_axis=input_owner_axis,
        ),
        jnp.zeros_like(x) if laplacian is None else laplacian,
    )


def make_local2_input(
    x: jnp.ndarray,
    *,
    blocks: jnp.ndarray,
    owners: OwnerRoles,
    input_shape: tuple[int, ...],
    input_owner_axis: int = 0,
    laplacian: jnp.ndarray | None = None,
) -> LapTuple:
    """Build an explicit Local2 Forward Laplacian test input."""
    return LapTuple(
        x,
        Local2Jacobian(
            blocks=blocks,
            owners=owners,
            input_shape=input_shape,
            input_owner_axis=input_owner_axis,
        ),
        jnp.zeros_like(x) if laplacian is None else laplacian,
    )


def tracked_case_input(
    x: jnp.ndarray,
    case: InputCase,
    key: int = 42,
    input_shape: tuple[int, ...] | None = None,
) -> jnp.ndarray | LapTuple:
    """Build one Forward Laplacian test input case.

    Local2 cases model a pairwise output with owner roles on the first two
    output axes, so their input shape requires equal first and second sizes.

    The default ``input_shape`` of ``(x.shape[0], 5)`` is intentionally
    independent of ``x``'s trailing shape: primitive tests only need a stable
    tracked coordinate basis, not production ``make_laplacian_input`` topology.
    """
    if input_shape is None:
        input_shape = (x.shape[0], 5)
    match case:
        case "plain":
            return x
        case "dense":
            dense_jacobian = jax.random.uniform(
                jax.random.key(key), (prod(input_shape), *x.shape), x.dtype
            )
            laplacian = jax.random.uniform(jax.random.key(key + 1), x.shape, x.dtype)
            return LapTuple(x, dense_jacobian, laplacian)
        case "local1":
            n = x.shape[0]
            blocks = jax.random.uniform(
                jax.random.key(key), (1, prod(input_shape[1:]), *x.shape), x.dtype
            )
            local1_jacobian = Local1Jacobian(
                blocks=blocks,
                owners=OwnerRoles(OwnerRole(axis=0, values=np.arange(n))),
                input_shape=input_shape,
                input_owner_axis=0,
            )
            laplacian = jax.random.uniform(jax.random.key(key + 1), x.shape, x.dtype)
            return LapTuple(x, local1_jacobian, laplacian)
        case "local2":
            if x.ndim < 2 or x.shape[0] != x.shape[1]:
                raise ValueError(
                    "Local2 primitive test inputs require equal first two dimensions."
                )
            n = x.shape[0]
            blocks = jax.random.uniform(
                jax.random.key(key), (2, prod(input_shape[1:]), *x.shape), x.dtype
            )
            local2_jacobian = Local2Jacobian(
                blocks=blocks,
                owners=OwnerRoles(
                    OwnerRole(axis=0, values=np.arange(n)),
                    OwnerRole(axis=1, values=np.arange(n)),
                ),
                input_shape=input_shape,
                input_owner_axis=0,
            )
            laplacian = jax.random.uniform(jax.random.key(key + 1), x.shape, x.dtype)
            return LapTuple(x, local2_jacobian, laplacian)
        case _:
            raise AssertionError(f"unknown input case {case!r}")


# Configurable sparse Jacobian fixtures shared by tests/laplacian/sparse. Keep
# operation-specific pathological cases beside their consuming test.


def sparse_local1_input(
    owner: OwnerRole,
    *,
    output_shape: tuple[int, ...] = (2, 1, 3),
    input_shape: tuple[int, ...] = (3, 1),
    key: int = 120,
    laplacian: jnp.ndarray | None = None,
    broadcast_blocks: bool = False,
) -> LapTuple:
    """Build a deterministic Local1 sparse Jacobian test input.

    Defaults model the common constant-owner case. Tests with a varying owner
    role, broadcast blocks, or a different rank should specify the relevant
    shape explicitly.
    """
    x = random_array("real", output_shape, key=key).astype(jnp.float32)
    n_support = prod(input_shape[1:]) if len(input_shape) > 1 else 1
    block_shape = (1, n_support, *x.shape)
    if broadcast_blocks:
        core_shape = (1, n_support, *((1,) * x.ndim))
        core = random_array("real", core_shape, key=key + 1).astype(jnp.float32)
        blocks = jnp.broadcast_to(core, block_shape)
    else:
        blocks = random_array("real", block_shape, key=key + 1).astype(jnp.float32)
    return make_local1_input(
        x,
        blocks=blocks,
        owners=OwnerRoles(owner),
        input_shape=input_shape,
        laplacian=laplacian,
    )


def sparse_local2_input(
    owners: OwnerRoles | None = None,
    *,
    output_shape: tuple[int, ...] = (4, 4, 3),
    input_shape: tuple[int, ...] = (4, 3),
    key: int = 200,
    laplacian: jnp.ndarray | None = None,
) -> LapTuple:
    """Build a deterministic Local2 sparse Jacobian test input."""
    if output_shape[0] != output_shape[1]:
        raise ValueError("Local2 sparse inputs require equal first two dimensions.")
    if owners is None:
        n = output_shape[0]
        owners = OwnerRoles(
            OwnerRole(0, np.arange(n, dtype=np.int32)),
            OwnerRole(1, np.arange(n, dtype=np.int32)),
        )
    x = random_array("real", output_shape, key=key).astype(jnp.float32)
    n_support = prod(input_shape[1:]) if len(input_shape) > 1 else 1
    blocks = random_array("real", (2, n_support, *x.shape), key=key + 1).astype(
        jnp.float32
    )
    return make_local2_input(
        x,
        blocks=blocks,
        owners=owners,
        input_shape=input_shape,
        laplacian=laplacian,
    )
