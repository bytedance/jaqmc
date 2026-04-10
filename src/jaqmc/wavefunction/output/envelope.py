# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import StrEnum

from flax import linen as nn
from jax import Array
from jax import numpy as jnp

__all__ = ["Envelope", "EnvelopeType"]

logger = logging.getLogger(__name__)


class EnvelopeType(StrEnum):
    r"""Envelope types controlling wavefunction decay near atoms.

    The envelope :math:`E_{ik} = \sum_I \pi_{kI} \exp(-d_{iIk})` modulates
    orbital output. The effective distance :math:`d_{iIk}` depends on the type.

    Attributes:
        isotropic: Scalar decay rate :math:`d = \sigma \| \mathbf{r} \|`.
            Simple but :math:`\sigma` can go negative.
        abs_isotropic: :math:`d = |\sigma| \| \mathbf{r} \|`.
            Always-decaying variant. Generally preferred.
        diagonal: Per-dimension decay rates
            :math:`d = \| \mathbf{r} \odot \boldsymbol{\sigma} \|`.
        null: No envelope (returns ones).
    """

    isotropic = "isotropic"
    abs_isotropic = "abs_isotropic"
    diagonal = "diagonal"
    null = "null"


class Envelope(nn.Module):
    r"""Envelope function for wavefunctions.

    Computes :math:`E_{ik} = \sum_I \pi_{kI} \exp(-d_{iIk})` where the
    effective distance :math:`d_{iIk}` depends on :class:`EnvelopeType`.
    Works for both OBC (molecules) and PBC (solids).

    Attributes:
        envelope_type: Type of envelope. See :class:`EnvelopeType`.
        ndets: Number of determinants.
        nspins: Tuple of (num_spin_up, num_spin_down) electrons.
        orbitals_spin_split: If True, use separate envelope parameters for
            each spin channel.
    """

    envelope_type: EnvelopeType
    ndets: int
    nspins: tuple[int, int]
    orbitals_spin_split: bool = False

    def setup(self):
        """Initialize envelope submodules based on the configuration."""
        n_up, n_down = self.nspins
        common = {"ndets": self.ndets}

        def make_module(n_orbitals: int) -> nn.Module | None:
            match self.envelope_type:
                case EnvelopeType.isotropic:
                    return _IsotropicEnvelope(
                        n_orbitals=n_orbitals, is_abs=False, **common
                    )
                case EnvelopeType.abs_isotropic:
                    return _IsotropicEnvelope(
                        n_orbitals=n_orbitals, is_abs=True, **common
                    )
                case EnvelopeType.diagonal:
                    return _DiagonalEnvelope(n_orbitals=n_orbitals, **common)
                case EnvelopeType.null:
                    return None
                case _:
                    raise ValueError(f"Unknown envelope: {self.envelope_type!r}")

        self._is_null = self.envelope_type == EnvelopeType.null

        if self.orbitals_spin_split and n_up > 0 and n_down > 0:
            self._spin_split = True
            # Each spin channel gets envelope for ALL orbitals
            n_total = sum(self.nspins)
            self._env_up = make_module(n_total)
            self._env_down = make_module(n_total)
        else:
            if self.orbitals_spin_split and self.is_initializing():
                logger.warning(
                    "orbitals_spin_split=True ignored: one spin channel is empty "
                    "(nspins=%s). Using shared envelope parameters.",
                    self.nspins,
                )
            self._spin_split = False
            self._env = make_module(sum(self.nspins))

    def __call__(self, ae_vectors: Array, r_ae: Array) -> Array:
        """Compute envelope values.

        Args:
            ae_vectors: Atom-electron displacement vectors (n_elec, n_atoms, ndim_feat).
                For OBC, ndim_feat=3 (Cartesian). For PBC, ndim_feat depends on
                the distance function (3 for nu_distance, 6 for tri_distance).
            r_ae: Atom-electron distances (n_elec, n_atoms).

        Returns:
            Envelope values of shape (ndets, n_elec, n_orbitals).
        """
        if self._is_null:
            n_elec = ae_vectors.shape[0]
            return jnp.ones((self.ndets, n_elec, n_elec))

        n_up, _ = self.nspins
        if self._spin_split:
            env_up = self._env_up(ae_vectors[:n_up], r_ae[:n_up])
            env_down = self._env_down(ae_vectors[n_up:], r_ae[n_up:])
            return jnp.concatenate([env_up, env_down], axis=1)
        else:
            return self._env(ae_vectors, r_ae)


class _IsotropicEnvelope(nn.Module):
    """Isotropic envelope with scalar sigma per atom-orbital."""

    n_orbitals: int
    ndets: int
    is_abs: bool = False

    @nn.compact
    def __call__(self, ae_vectors: Array, r_ae: Array) -> Array:
        _, n_atoms = r_ae.shape
        shape = (self.n_orbitals, n_atoms, self.ndets)
        pi = self.param("pi", nn.initializers.ones, shape)
        sigma = self.param("sigma", nn.initializers.ones, shape)

        r = r_ae[:, None, :, None]
        exponent = -jnp.abs(sigma * r) if self.is_abs else sigma * -r
        envelope = jnp.sum(pi * jnp.exp(exponent), axis=2)
        return jnp.transpose(envelope, (2, 0, 1))


class _DiagonalEnvelope(nn.Module):
    """Diagonal envelope with per-dimension decay rates."""

    n_orbitals: int
    ndets: int

    @nn.compact
    def __call__(self, ae_vectors: Array, r_ae: Array) -> Array:
        _, n_atoms = r_ae.shape
        ndim_feat = ae_vectors.shape[-1]
        pi_shape = (self.n_orbitals, n_atoms, self.ndets)
        sigma_shape = (self.n_orbitals, n_atoms, ndim_feat, self.ndets)
        pi = self.param("pi", nn.initializers.ones, pi_shape)
        sigma = self.param("sigma", nn.initializers.ones, sigma_shape)

        # ae_vectors: (n_elec, n_atoms, ndim_feat) -> (n_elec, 1, n_atoms, ndim_feat, 1)
        ae_expanded = ae_vectors[:, None, :, :, None]
        scaled = sigma * ae_expanded
        r_scaled = jnp.sqrt(jnp.sum(scaled**2, axis=3))
        envelope = jnp.sum(pi * jnp.exp(-r_scaled), axis=2)
        return jnp.transpose(envelope, (2, 0, 1))
