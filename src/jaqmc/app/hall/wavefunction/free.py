# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Free-electron monopole harmonics wavefunction (for benchmarking)."""

import numpy as np
from jax import numpy as jnp
from scipy import special as ss

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction


def make_monopole_harm(q: float, ell: float, m: float):
    r"""Create a monopole harmonic function :math:`Y_{q,\ell,m}`.

    Args:
        q: Monopole charge.
        ell: Angular momentum quantum number.
        m: Magnetic quantum number.

    Returns:
        A callable ``f(electrons) -> complex array``.
    """
    norm_factor = np.sqrt(
        ((2 * ell + 1) / (4 * np.pi))
        * (ss.factorial(ell - m) * ss.factorial(ell + m))
        / (ss.factorial(ell - q) * ss.factorial(ell + q))
    )
    s = np.arange(int(ell - m) + 1)
    sum_factors = jnp.array(
        (-1) ** (ell - m - s) * ss.comb(ell - q, s) * ss.comb(ell + q, ell - m - s)
    )

    def Y_qlm(electrons: jnp.ndarray) -> jnp.ndarray:
        theta, phi = electrons[..., 0], electrons[..., 1]
        x = jnp.cos(theta)
        theta_part = jnp.sum(
            sum_factors
            * (1 - x[..., None]) ** (ell - s - (m + q) / 2)
            * (1 + x[..., None]) ** (s + (m + q) / 2),
            axis=-1,
        )
        return norm_factor / 2**ell * theta_part * jnp.exp(1j * m * phi)

    return Y_qlm


class Free(Wavefunction[HallData, ComplexWFOutput]):
    """Free-electron wavefunction using monopole harmonics.

    Fills the lowest Landau level with monopole harmonics. Useful as a
    reference for verifying kinetic energy computations.

    Args:
        nspins: ``(n_up, n_down)`` electron counts.
        flux: Magnetic flux :math:`2Q`.
    """

    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()

    def setup(self):
        orbitals = []
        remaining_elec = sum(self.nspins)
        m = ell = q = self.flux / 2
        while remaining_elec > 0:
            orbitals.append(make_monopole_harm(q, ell, m))
            remaining_elec -= 1
            m -= 1
            if m < -ell:
                ell += 1
                m = ell
        self.orbital_fns = orbitals

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        orbitals = jnp.stack([phi(electrons) for phi in self.orbital_fns])
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)
        logpsi = jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax
        return ComplexWFOutput(logpsi=logpsi)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]
