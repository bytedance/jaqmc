# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exact Laughlin wavefunction on the Haldane sphere (for benchmarking)."""

from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction


class Laughlin(Wavefunction[HallData, ComplexWFOutput]):
    """Laughlin wavefunction for ground states on the Haldane sphere.

    Constructs the Laughlin state as a Slater determinant of composite
    fermion orbitals with an attached Jastrow factor.

    Args:
        nspins: ``(n_up, n_down)`` electron counts.
        flux: Magnetic flux :math:`2Q`.
        cf_flux: Composite fermion flux attachment parameter :math:`p`.
    """

    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()
    cf_flux: int = 1

    def setup(self):
        Q = self.flux / 2
        self.Q1 = Q - self.cf_flux * (sum(self.nspins) - 1)

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        theta, phi = electrons[..., 0], electrons[..., 1]
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]

        orbitals = self._full_orbitals(u, v)
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)
        logpsi = jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax
        return ComplexWFOutput(logpsi=logpsi)

    def _full_orbitals(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        Q = self.Q1
        m = jnp.arange(-Q, Q + 1)
        element = u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0])
        jastrow = jnp.prod(element, axis=-1, keepdims=True)
        return u ** (Q + m) * v ** (Q - m) * jastrow

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]
