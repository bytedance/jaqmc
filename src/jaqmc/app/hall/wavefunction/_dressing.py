# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Flux dressing blocks for neural Hall wavefunctions.

Learnable dressings multiply each orbital by pair factors
:math:`\prod_{c}\prod_{j\ne i} z_i^\mathsf{T} N z_j`, with
:math:`N = N_0 + \Delta` and :math:`N_0` the antisymmetric spinor matrix, so
every variant reduces to :math:`u_i v_j - v_i u_j` at initialization.
``type`` selects a fixed Laughlin factor or a learnable full pair-spinor
matrix, ``conditioning`` whether :math:`\Delta` is constant or read out from
backbone features, ``orbital_dependent`` whether each orbital has its own
:math:`\Delta`, and ``det_dependent`` whether each determinant does.
"""

from typing import Literal

from flax import linen as nn
from jax import numpy as jnp

from jaqmc.utils.config import configurable_dataclass

from ._pair_factor import log_pair_factor

PAIR_SPINOR_MATRIX = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]])


@configurable_dataclass(frozen=True)
class DressingConfig:
    """Configuration for the flux dressing applied to Hall orbitals.

    ``laughlin`` ignores ``conditioning``, ``orbital_dependent``, and
    ``det_dependent``.
    """

    type: Literal["laughlin", "full"] = "full"
    degree: int = 0
    conditioning: Literal["backbone", "constant"] = "backbone"
    orbital_dependent: bool = True
    det_dependent: bool = True

    def __post_init__(self) -> None:
        if self.type not in {"laughlin", "full"}:
            raise ValueError(f"Unsupported dressing type: {self.type!r}.")
        if self.conditioning not in {"backbone", "constant"}:
            raise ValueError(
                f"Unsupported dressing conditioning: {self.conditioning!r}."
            )
        if self.degree < 0:
            raise ValueError("dressing degree must be nonnegative.")
        if self.type != "laughlin" and self.degree % 2:
            raise ValueError("full dressing requires an even degree.")


def reduced_monopole_strength(*, flux: int, degree: int, nelec: int) -> float:
    r"""Return :math:`Q^* = (\mathrm{flux} - p(N-1)) / 2`."""
    return (flux - degree * (nelec - 1)) / 2


def _laughlin_dressing(
    orbitals: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, degree: int
) -> jnp.ndarray:
    nelec = u.shape[0]
    element = u * v[..., None] - u[..., None] * v + jnp.eye(nelec)
    log_factor = jnp.sum(jnp.triu(jnp.log(element), k=1)) * degree
    return jnp.exp(log_factor / nelec) * orbitals


class Dressing(nn.Module):
    """Apply fixed Laughlin or learned pair-spinor dressing factors."""

    config: DressingConfig
    ndets: int

    @nn.compact
    def __call__(
        self,
        orbitals: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        h_one: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply the configured dressing to determinant-leading orbitals.

        Args:
            orbitals: Determinant-leading orbital matrices.
            u: First spinor coordinate for every electron.
            v: Second spinor coordinate for every electron.
            h_one: Backbone features for every electron.

        Returns:
            Determinant-leading orbitals with the dressing applied.
        """
        cfg = self.config
        if cfg.degree == 0:
            return orbitals
        if cfg.type == "laughlin":
            return _laughlin_dressing(orbitals, u, v, cfg.degree)

        ndets = self.ndets if cfg.det_dependent else 1
        norb = u.shape[0] if cfg.orbital_dependent else 1
        shape = (ndets, cfg.degree // 2, norb)
        log_factor = self._full(u, v, h_one, shape)
        return orbitals * jnp.moveaxis(jnp.exp(log_factor), 1, 2)

    def _readout(self, h_one: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
        """Return a zero-initialized complex readout with shape ``(nelec, *shape)``."""
        out = nn.DenseGeneral(
            features=(*shape, 2),
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h_one)
        return out[..., 0] + 1j * out[..., 1]

    def _full(
        self,
        u: jnp.ndarray,
        v: jnp.ndarray,
        h_one: jnp.ndarray,
        shape: tuple[int, int, int],
    ) -> jnp.ndarray:
        r"""Reduce pair factors :math:`z_i^\mathsf{T} N z_j` with a full 2x2 matrix.

        Returns:
            Log pair factors with shape ``(ndets, norb, nelec)``.
        """
        z = jnp.stack((u, v), axis=-1)
        if self.config.conditioning == "constant":
            matrix_real = self.param(
                "matrix_real",
                nn.initializers.constant(PAIR_SPINOR_MATRIX),
                (*shape, 2, 2),
            )
            matrix_imag = self.param(
                "matrix_imag", nn.initializers.zeros, (*shape, 2, 2)
            )
            # Contract real and imaginary parts separately; combining them first
            # lets KFAC match a scale-and-shift block on the constant ``1j``.
            w = jnp.einsum("ia,dckab->idckb", z, matrix_real)
            w = w + 1j * jnp.einsum("ia,dckab->idckb", z, matrix_imag)
            return log_pair_factor(w, z)
        n_mat = PAIR_SPINOR_MATRIX + self._readout(h_one, (*shape, 2, 2))
        return log_pair_factor(jnp.einsum("ia,idckab->idckb", z, n_mat), z)
