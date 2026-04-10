# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jax import numpy as jnp


def pair_displacements_within(
    positions: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes pairwise displacements and distances within one species.

    Args:
        positions: Particle positions with shape (n_particles, ndim).

    Returns:
        A tuple ``(disp, r)``:

        - **disp** -- Pairwise displacement vectors ``r_i - r_j`` with shape
          ``(n_particles, n_particles, ndim)``.
        - **r** -- Pairwise distances with shape ``(n_particles, n_particles)``.

    Raises:
        ValueError: If ``positions`` does not have shape ``(n_particles, ndim)``.

    Examples:
        >>> from jax import numpy as jnp
        >>> pos = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        >>> disp, r = pair_displacements_within(pos)
        >>> disp.shape
        (3, 3, 2)
        >>> r.shape
        (3, 3)
    """
    if positions.ndim != 2:
        raise ValueError(
            f"Expected positions of shape (n_particles, ndim), got {positions.shape}."
        )

    disp = positions[:, None, :] - positions

    # Avoid computing the norm of zero (undefined gradient at ||0||) by
    # shifting the diagonal entries before taking the norm and then masking
    # them back out.
    n = disp.shape[0]
    eye = jnp.eye(n, dtype=positions.dtype)
    disp_for_norm = disp + eye[..., None]
    r = jnp.linalg.norm(disp_for_norm, axis=-1) * (1.0 - eye)
    return disp, r


def pair_displacements_between(
    positions_a: jnp.ndarray, positions_b: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes pairwise displacements and distances across two species.

    Args:
        positions_a: Positions of the first species, shape (n_a, ndim).
        positions_b: Positions of the second species, shape (n_b, ndim).

    Returns:
        A tuple ``(disp, r)``:

        - **disp** -- Pairwise displacement vectors ``r_a - r_b`` with shape
          ``(n_a, n_b, ndim)``.
        - **r** -- Pairwise distances with shape ``(n_a, n_b)``.

    Raises:
        ValueError: If inputs are not two-dimensional or if their spatial
            dimensions (``ndim``) do not match.
    """
    if positions_a.ndim != 2:
        raise ValueError(
            f"Expected positions_a of shape (n_a, ndim), got {positions_a.shape}."
        )
    if positions_b.ndim != 2:
        raise ValueError(
            f"Expected positions_b of shape (n_b, ndim), got {positions_b.shape}."
        )
    if positions_a.shape[1] != positions_b.shape[1]:
        raise ValueError(
            "The spatial dimensions of positions_a and positions_b do not match. "
            f"Got positions_a.shape[1]={positions_a.shape[1]} and "
            f"positions_b.shape[1]={positions_b.shape[1]}."
        )

    disp = positions_a[:, None, :] - positions_b
    r = jnp.linalg.norm(disp, axis=-1)
    return disp, r
