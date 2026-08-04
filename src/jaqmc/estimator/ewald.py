# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

import jax
from jax import numpy as jnp

from jaqmc.geometry import pbc

logger = logging.getLogger(__name__)

# Reciprocal vectors whose Ewald weight falls below this threshold are dropped.
_GWEIGHT_TOL = 1e-12


def _halfspace_gpoint_indices(dim: int, ewald_gmax: int) -> jnp.ndarray:
    r"""Integer reciprocal-lattice indices covering half of reciprocal space.

    Enumerates every nonzero integer index vector whose first nonzero component
    is positive, so that :math:`\mathbf G` and :math:`-\mathbf G` are not both
    included (the Ewald reciprocal sum is symmetric under :math:`\mathbf G \to
    -\mathbf G`). The half-space is swept as ``dim`` disjoint blocks: the
    ``lead``-th block fixes the first ``lead`` components to zero, takes the
    next component in ``[1, ewald_gmax]``, and lets the rest range over
    ``[-ewald_gmax, ewald_gmax]``.

    Args:
        dim: Spatial dimension (2 or 3).
        ewald_gmax: Reciprocal-space cutoff (indices per direction).

    Returns:
        Integer array with shape ``(npoints, dim)``.
    """
    full = jnp.arange(-ewald_gmax, ewald_gmax + 1)
    positive = jnp.arange(1, ewald_gmax + 1)
    zero = jnp.asarray([0])
    blocks = []
    for lead in range(dim):
        axes = [zero] * lead + [positive] + [full] * (dim - lead - 1)
        grid = jnp.meshgrid(*axes, indexing="ij")
        blocks.append(jnp.stack([g.reshape(-1) for g in grid], axis=-1))
    return jnp.concatenate(blocks, axis=0)


class _EwaldSumBase:
    r"""Shared Ewald summation for periodic electrostatics in 2D or 3D.

    Decomposes the Coulomb interaction into rapidly converging real-space and
    reciprocal-space series:

    .. math::
        V_{\text{Ewald}} = V_{\text{real}} + V_{\text{recip}} +
        V_{\text{self}} + V_{\text{charged}}

    All charged particles (electrons and ions) are treated uniformly as point
    charges. Subclasses fix the dimension and supply the dimension-specific
    reciprocal-space weight and charged-background constant; everything else --
    the real-space sum, structure factor, self-energy, minimum-image distances
    and Ewald parameter -- is shared.

    .. seealso:: :doc:`/guide/estimators/ewald` for the full formulation
       and implementation notes.
    """

    dim: int
    cell_measure_name: str

    def _init_common(self, lattice: jnp.ndarray, ewald_gmax: int, nlatvec: int) -> None:
        """Store the lattice and precompute real- and reciprocal-space data.

        Args:
            lattice: (dim, dim) matrix of simulation-cell lattice vectors.
            ewald_gmax: Cutoff for the reciprocal space sum (number of
                G-vectors in each direction).
            nlatvec: Cutoff for the real space sum (number of periodic images
                in each direction).
        """
        self.latvec = jnp.asarray(lattice)
        self.dist = pbc.build_distance_fn(self.latvec)
        self.set_lattice_displacements(nlatvec)
        self.set_up_reciprocal_ewald_sum(ewald_gmax)

    def set_lattice_displacements(self, nlatvec: int) -> None:
        """Generates lattice-vector displacements for the real-space sum."""
        grid = jnp.meshgrid(
            *[jnp.arange(-nlatvec, nlatvec + 1)] * self.dim, indexing="ij"
        )
        indices = jnp.stack(grid, axis=-1).reshape((-1, self.dim))
        self.lattice_displacements = indices @ self.latvec

    def set_up_reciprocal_ewald_sum(self, ewald_gmax: int) -> None:
        r"""Select G-vectors and precompute the Ewald constants.

        Sets the following attributes:

        - ``alpha``: Ewald separation parameter, :math:`5 / h_{\min}` where
          :math:`h_{\min}` is the smallest perpendicular cell height.
        - ``gpoints``, ``gweight``: Selected reciprocal lattice vectors and
          their weights (see :meth:`_reciprocal_weight`).
        - ``ijconst``: Charged-system background correction factor.
        - ``self_const_factor``: Self-energy correction :math:`-\alpha /
          \sqrt{\pi}`.

        Args:
            ewald_gmax: Reciprocal-space cutoff (G-vectors per direction).

        Raises:
            ValueError: If the lattice has a non-finite or zero cell measure.
        """
        cell_measure = jnp.abs(jnp.linalg.det(self.latvec))
        if not bool(jnp.isfinite(cell_measure) & (cell_measure > 0.0)):
            raise ValueError(
                f"{self.dim}D Ewald lattice must have a finite, non-zero "
                f"{self.cell_measure_name}; got {cell_measure}."
            )
        recvec = jnp.linalg.inv(self.latvec).T

        smallestheight = jnp.amin(1.0 / jnp.linalg.norm(recvec, axis=1))
        self.alpha = 5.0 / smallestheight
        logger.info("Setting Ewald alpha to %s", self.alpha)

        indices = _halfspace_gpoint_indices(self.dim, ewald_gmax)
        gpoints = indices @ recvec * (2.0 * jnp.pi)
        gweight = self._reciprocal_weight(gpoints, cell_measure)
        keep = gweight > _GWEIGHT_TOL
        self.gpoints, self.gweight = gpoints[keep], gweight[keep]

        self.ijconst = self._background_const(cell_measure)
        # Self-interaction constant factor per particle charge^2.
        self.self_const_factor = -self.alpha / jnp.sqrt(jnp.pi)

    def _reciprocal_weight(
        self, gpoints: jnp.ndarray, cell_measure: jnp.ndarray
    ) -> jnp.ndarray:
        r"""Per-G-vector reciprocal-space weight (dimension specific).

        Args:
            gpoints: Reciprocal lattice vectors, shape ``(npoints, dim)``.
            cell_measure: Cell volume (3D) or area (2D).

        Returns:
            Weight :math:`W(\mathbf G)` for each reciprocal vector, shape
            ``(npoints,)``.
        """
        raise NotImplementedError

    def _background_const(self, cell_measure: jnp.ndarray) -> jnp.ndarray:
        """Charged-system background correction factor (dimension specific).

        Args:
            cell_measure: Cell volume (3D) or area (2D).

        Returns:
            Scalar background correction factor ``ijconst``.
        """
        raise NotImplementedError

    def energy(self, coords: jnp.ndarray, charges: jnp.ndarray) -> jnp.ndarray:
        r"""Total electrostatic energy of a periodic system of point charges.

        Treats all particles (electrons and ions) uniformly and sums the four
        Ewald components:

        .. math::
            E_{\text{total}} = E_{\text{real}} + E_{\text{recip}} +
            E_{\text{self}} + E_{\text{charged}}

        Args:
            coords: Particle coordinates ``(N, dim)``.
            charges: Particle charges ``(N,)``.

        Returns:
            Total electrostatic energy.
        """
        # 1. Real-space sum over minimum-image displacements and lattice images.
        displacements, _ = self.dist(coords, coords)
        rvec = displacements[None, ...] + self.lattice_displacements[:, None, None, :]
        r = jnp.linalg.norm(rvec, axis=-1)
        charge_prod = charges[:, None] * charges[None, :]

        # Mask out self-interaction (i=j in the central image n=0).
        center_image_idx = jnp.argmin(
            jnp.linalg.norm(self.lattice_displacements, axis=-1)
        )
        n_imgs = self.lattice_displacements.shape[0]
        n_parts = coords.shape[0]
        mask = jnp.ones((n_imgs, n_parts, n_parts))
        mask = mask.at[center_image_idx].set(1.0 - jnp.eye(n_parts))

        # Neutralize masked terms and guard against coincident particles so the
        # masked-out infinities never turn into NaNs.
        r_safe = jnp.where(mask < 0.5, 1.0, jnp.maximum(r, 1e-7))
        pot_term = jax.lax.erfc(self.alpha * r_safe) / r_safe
        v_real = 0.5 * jnp.sum(charge_prod[None, :, :] * pot_term * mask)

        # 2. Reciprocal-space sum via the structure factor.
        structure_factor = jnp.dot(jnp.exp(1j * (self.gpoints @ coords.T)), charges)
        v_recip = jnp.dot(self.gweight, jnp.abs(structure_factor) ** 2)

        # 3. Self-energy correction (removes each charge's own Gaussian cloud).
        v_self = self.self_const_factor * jnp.sum(charges**2)

        # 4. Charged-system background correction.
        total_charge = jnp.sum(charges)
        v_charged = 0.5 * self.ijconst * total_charge**2

        return v_real + v_recip + v_self + v_charged


class EwaldSum(_EwaldSumBase):
    r"""Three-dimensional Ewald summation for electrostatic energy.

    .. seealso:: :doc:`/guide/estimators/ewald` for the full formulation
       and implementation notes.

    Args:
        supercell_lattice: (3, 3) matrix representing the supercell lattice vectors.
        ewald_gmax: Cutoff for the reciprocal space sum (number of G-vectors in
            each direction). Determines accuracy of :math:`V_{\text{recip}}`.
        nlatvec: Cutoff for the real space sum (number of periodic images in each
            direction). Determines accuracy of :math:`V_{\text{real}}`.
    """

    dim = 3
    cell_measure_name = "volume"

    def __init__(
        self,
        supercell_lattice: jnp.ndarray,
        ewald_gmax: int = 200,
        nlatvec: int = 1,
    ):
        """Initialize EwaldSum."""
        self._init_common(supercell_lattice, ewald_gmax, nlatvec)

    def _reciprocal_weight(
        self, gpoints: jnp.ndarray, cell_measure: jnp.ndarray
    ) -> jnp.ndarray:
        r"""Weight :math:`\frac{4\pi}{\Omega G^2} e^{-G^2/4\alpha^2}`.

        Returns:
            Reciprocal-space weight for each G-vector.
        """
        gsquared = jnp.sum(gpoints**2, axis=-1)
        return (
            4.0
            * jnp.pi
            * jnp.exp(-gsquared / (4.0 * self.alpha**2))
            / (cell_measure * gsquared)
        )

    def _background_const(self, cell_measure: jnp.ndarray) -> jnp.ndarray:
        r"""Charged-background factor :math:`-\pi / (\Omega \alpha^2)`.

        Returns:
            Scalar charged-background correction factor.
        """
        return -jnp.pi / (cell_measure * self.alpha**2)


class EwaldSum2D(_EwaldSumBase):
    r"""Two-dimensional Ewald summation for moire-style Coulomb energies.

    Args:
        lattice: (2, 2) matrix representing the simulation-cell lattice vectors.
        ewald_gmax: Cutoff for the reciprocal space sum.
        nlatvec: Cutoff for the real space sum.
    """

    dim = 2
    cell_measure_name = "area"

    def __init__(
        self,
        lattice: jnp.ndarray,
        *,
        ewald_gmax: int = 200,
        nlatvec: int = 1,
    ):
        lattice = jnp.asarray(lattice)
        if lattice.shape != (2, 2):
            raise ValueError(f"Expected a 2x2 lattice, got {lattice.shape}.")
        self._init_common(lattice, ewald_gmax, nlatvec)

    def _reciprocal_weight(
        self, gpoints: jnp.ndarray, cell_measure: jnp.ndarray
    ) -> jnp.ndarray:
        r"""Weight :math:`\frac{2\pi}{A\,G}\,\mathrm{erfc}(G/2\alpha)`.

        Returns:
            Reciprocal-space weight for each G-vector.
        """
        gnorm = jnp.linalg.norm(gpoints, axis=-1)
        return (
            2.0
            * jnp.pi
            * jax.lax.erfc(gnorm / (2.0 * self.alpha))
            / (cell_measure * gnorm)
        )

    def _background_const(self, cell_measure: jnp.ndarray) -> jnp.ndarray:
        r"""Charged-background factor :math:`-2\sqrt{\pi} / (A \alpha)`.

        Returns:
            Scalar charged-background correction factor.
        """
        return -2.0 * jnp.sqrt(jnp.pi) / (cell_measure * self.alpha)
