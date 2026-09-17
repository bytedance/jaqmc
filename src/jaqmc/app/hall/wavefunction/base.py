# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared Hall-sphere wavefunction adapter for spinor evaluation."""

from abc import ABC, abstractmethod

from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.geometry.sphere import spinor_coordinates_from_angles
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction

__all__ = ["HallWavefunction", "lll_monopole_harmonics"]


def lll_monopole_harmonics(u: jnp.ndarray, v: jnp.ndarray, Q: float) -> jnp.ndarray:
    """Unnormalized LLL monopole harmonics ``u**a * v**(2Q-a)``, ``a=0..2Q``.

    Use integer powers to stay finite at the poles.

    Returns:
        Basis with shape ``[..., 2Q + 1]``.
    """
    n_orb = round(2 * Q + 1)
    return jnp.stack([(u**a) * (v ** (n_orb - 1 - a)) for a in range(n_orb)], axis=-1)


class HallWavefunction(Wavefunction[HallData, ComplexWFOutput], ABC):
    """Hall-sphere wavefunction evaluated from a monopole spinor.

    Subclasses implement :meth:`_logpsi_from_spinor`. ``__call__`` converts
    sampled ``(theta, phi)`` to ``(u, v)``.
    """

    def __call__(self, data: HallData) -> ComplexWFOutput:
        u, v = spinor_coordinates_from_angles(data.electrons)
        return ComplexWFOutput(logpsi=self._logpsi_from_spinor(u, v))

    @abstractmethod
    def _logpsi_from_spinor(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log psi from a monopole spinor."""

    def logpsi_from_spinor(
        self,
        params: Params,
        u: jnp.ndarray,
        v: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return log psi from a local monopole spinor."""
        return self.apply(params, u, v, method=self._logpsi_from_spinor)  # type: ignore

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]
