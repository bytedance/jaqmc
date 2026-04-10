# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from flax import linen as nn
from jax import numpy as jnp


class SimpleEEJastrow(nn.Module):
    r"""Jastrow factor for electron-electron cusp conditions.

    Implements the simple electron-electron Jastrow factor from FermiNet:

    .. math::

        J = \sum_{i<j} f(r_{ij})

    where the cusp function is:

    .. math::

        f(r) = -\frac{c \alpha^2}{\alpha + r}

    with :math:`c = 0.25` for parallel spins and :math:`c = 0.5` for
    antiparallel spins.

    This form satisfies the electron-electron cusp condition:

    .. math::

        \frac{d f}{d r}\bigg|_{r=0} = \frac{c \alpha^2}{\alpha^2} = c

    The Jastrow factor is applied multiplicatively to orbitals as:

    .. math::

        \phi'_{ij} = \phi_{ij} \cdot \exp(J / n_{electrons})

    Args:
        nspins: Tuple of (num_spin_up, num_spin_down) electrons.
        alpha_init: Initial value for the learnable decay parameters
            (alpha_par and alpha_anti). Defaults to 1.0.
    """

    nspins: tuple[int, int]
    alpha_init: float = 1.0

    @nn.compact
    def __call__(self, r_ee: jnp.ndarray) -> jnp.ndarray:
        """Compute the Jastrow factor from electron-electron distances.

        Args:
            r_ee: Electron-electron distances of shape ``(n_electrons, n_electrons)``.
                The diagonal elements (self-distances) should be zero.

        Returns:
            Scalar Jastrow factor value (sum over all unique pairs).
        """
        n_up, n_down = self.nspins

        # Learnable decay parameters (initialized to alpha_init)
        alpha_initializer = nn.initializers.constant(self.alpha_init)
        alpha_par = self.param("alpha_par", alpha_initializer, (1,))
        alpha_anti = self.param("alpha_anti", alpha_initializer, (1,))

        # Split distance matrix by spin blocks
        # r_ee has shape (n_electrons, n_electrons) where n_electrons = n_up + n_down
        # We need to extract:
        #   - up-up block: r_ee[:n_up, :n_up]
        #   - down-down block: r_ee[n_up:, n_up:]
        #   - up-down block: r_ee[:n_up, n_up:] (antiparallel)

        # Extract upper triangular indices for parallel spin pairs
        # (avoiding double counting and diagonal)
        r_up_up = r_ee[:n_up, :n_up]
        r_down_down = r_ee[n_up:, n_up:]

        # Parallel spin pairs (same spin)
        parallel_pairs = []
        if n_up > 1:
            # Upper triangular indices for up-up block
            idx_up = jnp.triu_indices(n_up, k=1)
            parallel_pairs.append(r_up_up[idx_up])
        if n_down > 1:
            # Upper triangular indices for down-down block
            idx_down = jnp.triu_indices(n_down, k=1)
            parallel_pairs.append(r_down_down[idx_down])

        # Antiparallel spin pairs (up-down)
        r_up_down = r_ee[:n_up, n_up:]

        # Compute Jastrow contributions
        jastrow_par = jnp.asarray(0.0)
        jastrow_anti = jnp.asarray(0.0)

        if parallel_pairs:
            r_par = jnp.concatenate(parallel_pairs)
            jastrow_par = jnp.sum(_cusp_function(r_par, 0.25, alpha_par))

        if n_up > 0 and n_down > 0:
            # All up-down pairs (no need for triangular indexing)
            r_anti = r_up_down.ravel()
            jastrow_anti = jnp.sum(_cusp_function(r_anti, 0.5, alpha_anti))

        return jastrow_par + jastrow_anti


def _cusp_function(r: jnp.ndarray, c: float, alpha: jnp.ndarray) -> jnp.ndarray:
    r"""Jastrow cusp function satisfying the electron cusp condition.

    Computes :math:`f(r) = -\frac{c \alpha^2}{\alpha + r}`.

    Args:
        r: Electron-electron distances. Can be empty (shape ``(0,)``).
        c: Cusp value (0.25 for parallel, 0.5 for antiparallel spins).
        alpha: Learnable decay parameter.

    Returns:
        Jastrow contribution for each distance. For empty input ``r``,
        returns an empty array; when summed, this contributes 0.0 to
        the total Jastrow factor.
    """
    return -(c * alpha**2) / (alpha + r)
