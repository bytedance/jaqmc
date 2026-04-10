# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

import jax
from jax import numpy as jnp

from jaqmc.geometry import pbc

logger = logging.getLogger(__name__)


class EwaldSum:
    r"""Ewald summation for electrostatic energy in periodic systems.

    Decomposes the Coulomb interaction into real-space and reciprocal-space
    series for rapid convergence:

    .. math::
        V_{\text{Ewald}} = V_{\text{real}} + V_{\text{recip}} +
        V_{\text{self}} + V_{\text{charged}}

    All charged particles (electrons and ions) are treated uniformly.

    .. seealso:: :doc:`/guide/estimators/ewald` for the full formulation
       and implementation notes.

    Args:
        supercell_lattice: (3, 3) matrix representing the supercell lattice vectors.
        ewald_gmax: Cutoff for the reciprocal space sum (number of G-vectors in
            each direction). Determines accuracy of :math:`V_{\text{recip}}`.
        nlatvec: Cutoff for the real space sum (number of periodic images in each
            direction). Determines accuracy of :math:`V_{\text{real}}`.
    """

    def __init__(
        self,
        supercell_lattice: jnp.ndarray,
        ewald_gmax: int = 200,
        nlatvec: int = 1,
    ):
        """Initialize EwaldSum."""
        self.dim = 3
        self.latvec = supercell_lattice
        self.dist = pbc.build_distance_fn(self.latvec)
        self.set_lattice_displacements(nlatvec)
        self.set_up_reciprocal_ewald_sum(ewald_gmax)

    def set_lattice_displacements(self, nlatvec):
        """Generates lattice-vector displacements for real-space sum."""
        XYZ = jnp.meshgrid(
            *[jnp.arange(-nlatvec, nlatvec + 1)] * self.dim, indexing="ij"
        )
        xyz = jnp.stack(XYZ, axis=-1).reshape((-1, self.dim))
        self.lattice_displacements = jnp.asarray(jnp.dot(xyz, self.latvec))

    def set_up_reciprocal_ewald_sum(self, ewald_gmax):
        r"""Initialize Ewald parameter, select G-vectors, and precompute constants.

        Sets the following attributes:

        - ``alpha``: Ewald separation parameter, :math:`5.0 / h_{\min}` where
          :math:`h_{\min}` is the smallest perpendicular cell height.
        - ``gpoints``, ``gweight``: Selected reciprocal lattice vectors and their
          weights :math:`W(G) = \frac{4\pi}{\Omega G^2} e^{-G^2/4\alpha^2}`.
        - ``ijconst``: Charged-system correction factor
          :math:`-\pi / (\Omega \alpha^2)`.
        - ``self_const_factor``: Self-energy correction factor
          :math:`-\alpha / \sqrt{\pi}`.

        Args:
            ewald_gmax: Cutoff for the reciprocal space sum (number of
                G-vectors in each direction).
        """
        cellvolume = jnp.linalg.det(self.latvec)
        recvec = jnp.linalg.inv(self.latvec).T

        # Determine alpha
        smallestheight = jnp.amin(1 / jnp.linalg.norm(recvec, axis=1))
        self.alpha = 5.0 / smallestheight
        logger.info("Setting Ewald alpha to %s", self.alpha)

        if self.dim == 3:
            gptsXpos = jnp.meshgrid(
                jnp.arange(1, ewald_gmax + 1),
                jnp.arange(-ewald_gmax, ewald_gmax + 1),
                jnp.arange(-ewald_gmax, ewald_gmax + 1),
                indexing="ij",
            )
            zero = jnp.asarray([0])
            gptsX0Ypos = jnp.meshgrid(
                zero,
                jnp.arange(1, ewald_gmax + 1),
                jnp.arange(-ewald_gmax, ewald_gmax + 1),
                indexing="ij",
            )
            gptsX0Y0Zpos = jnp.meshgrid(
                zero, zero, jnp.arange(1, ewald_gmax + 1), indexing="ij"
            )
            pos_list = [gptsXpos, gptsX0Ypos, gptsX0Y0Zpos]
            gs = zip(
                *[select_big_3d(x, cellvolume, recvec, self.alpha) for x in pos_list]
            )
            self.gpoints, self.gweight = [jnp.concatenate(x, axis=0) for x in gs]

        # Precompute constants for background correction
        self.ijconst = -jnp.pi / (cellvolume * self.alpha**2)
        # Self-interaction constant factor per particle charge^2
        self.self_const_factor = -self.alpha / jnp.sqrt(jnp.pi)

    def energy(self, coords: jnp.ndarray, charges: jnp.ndarray) -> jnp.ndarray:
        r"""Calculates the total electrostatic energy for a general system.

        This method implements a unified Ewald summation where all particles (electrons
        and ions) are treated as point charges. It computes the total energy as a sum
        of four components:

        .. math::
            E_{\text{total}} = E_{\text{real}} + E_{\text{recip}} +
            E_{\text{self}} + E_{\text{charged}}

        Args:
            coords: Particle coordinates (N, 3).
            charges: Particle charges (N,).

        Returns:
            Total electrostatic energy.
        """
        # 1. Real space sum
        # displacements shape: (N, N, 3)
        displacements, _ = self.dist(coords, coords)

        # Add lattice images: (L, N, N, 3)
        rvec = displacements[None, ...] + self.lattice_displacements[:, None, None, :]
        r = jnp.linalg.norm(rvec, axis=-1)

        # Charge product matrix (N, N)
        charge_prod = charges[:, None] * charges[None, :]

        # Zero out the self-interaction terms (i=j and n=0)
        # n=0 is the image where shift is zero.
        center_image_idx = jnp.argmin(
            jnp.linalg.norm(self.lattice_displacements, axis=-1)
        )

        # Mask: 1 everywhere, 0 at (n=center, i=i, j=i)
        n_imgs = self.lattice_displacements.shape[0]
        n_parts = coords.shape[0]

        mask = jnp.ones((n_imgs, n_parts, n_parts))
        mask = mask.at[center_image_idx].set(1.0 - jnp.eye(n_parts))

        # Although we mask out self-interactions by multiplying zero, having infinities
        # in potential energy terms can result in NaN. Here we make sure `r` is finite.
        r_safe = jnp.where(r < 1e-7, 1e-7, r)
        pot_term = jax.lax.erfc(self.alpha * r_safe) / r_safe

        v_real = 0.5 * jnp.sum(charge_prod[None, :, :] * pot_term * mask)

        # 2. Reciprocal space sum
        GdotR = jnp.dot(self.gpoints, coords.T)
        structure_factor = jnp.dot(jnp.exp(1j * GdotR), charges)
        v_recip = jnp.dot(self.gweight, jnp.abs(structure_factor) ** 2)

        # Self-energy correction (removes self-interaction of Gaussian clouds)
        v_self = self.self_const_factor * jnp.sum(charges**2)

        # Charged-system background correction
        total_charge = jnp.sum(charges)
        v_charged = 0.5 * self.ijconst * total_charge**2

        return v_real + v_recip + v_self + v_charged


def select_big_3d(gpts, cellvolume, recvec, alpha, tol=1e-12):
    """Selects G-vectors with significant contributions to the Ewald sum.

    Selects G-vectors whose squared norm is less than a cutoff determined by
    ``alpha`` and ``tol``.

    Args:
        gpts: Tuple of meshgrid arrays representing G-vectors.
        cellvolume: Volume of the simulation cell.
        recvec: Reciprocal lattice vectors.
        alpha: Ewald separation parameter.
        tol: Error tolerance for selecting G-vectors.

    Returns:
        A tuple ``(gpoints, gweight)`` containing selected G-vectors and their weights.
    """
    if isinstance(gpts, (tuple, list)):
        gpts = jnp.stack(gpts, axis=0)

    gpoints = jnp.einsum("j...,jk->...k", gpts, recvec) * 2 * jnp.pi
    gsquared = jnp.einsum("...k,...k->...", gpoints, gpoints)
    gweight = 4 * jnp.pi * jnp.exp(-gsquared / (4 * alpha**2))
    gweight /= cellvolume * gsquared
    bigweight = gweight > tol
    return gpoints[bigweight], gweight[bigweight]
