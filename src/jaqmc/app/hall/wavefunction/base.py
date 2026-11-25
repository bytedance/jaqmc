# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared Hall-sphere wavefunction adapter for spinor evaluation."""

from abc import ABC, abstractmethod

from flax import linen as nn
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.geometry.sphere import spinor_coordinates_from_angles
from jaqmc.utils.wiring import runtime_dep
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

    Args:
        nspins: ``(n_up, n_down)`` electron counts (runtime dep).
        flux: Magnetic flux :math:`2Q` (runtime dep).
        interaction_strength: Coulomb interaction scaling factor (runtime dep).
        radius: Sphere radius (runtime dep).
    """

    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()
    interaction_strength: float = runtime_dep()
    radius: float = runtime_dep()

    def __call__(self, data: HallData) -> ComplexWFOutput:
        u, v = spinor_coordinates_from_angles(data.electrons)
        return ComplexWFOutput(logpsi=self._logpsi_from_spinor(u, v))

    @abstractmethod
    def _logpsi_from_spinor(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Evaluate log psi from a monopole spinor."""

    @nn.nowrap
    def logpsi_from_spinor(
        self,
        params: Params,
        u: jnp.ndarray,
        v: jnp.ndarray,
        *,
        precision: str | None = None,
    ) -> jnp.ndarray:
        """Return log psi from a local monopole spinor.

        Raises:
            ValueError: If an explicit precision mode is requested.
        """
        if precision is not None:
            raise ValueError(
                f"{type(self).__name__} does not support precision mode {precision!r}."
            )
        return self.apply(params, u, v, method=self._logpsi_from_spinor)  # type: ignore

    @nn.nowrap
    def logpsi(
        self, params: Params, data: HallData, *, precision: str | None = None
    ) -> jnp.ndarray:
        return self.evaluate(params, data, precision=precision)["logpsi"]

    @nn.nowrap
    def phase_logpsi(
        self, params: Params, data: HallData, *, precision: str | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logpsi = self.logpsi(params, data, precision=precision)
        return jnp.exp(1j * logpsi.imag), logpsi.real
