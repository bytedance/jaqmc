# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Free-electron monopole harmonics wavefunction (for benchmarking)."""

import numpy as np
from jax import numpy as jnp
from scipy import special as ss

from jaqmc.utils.wiring import runtime_dep

from .base import HallWavefunction

__all__ = ["Free"]


def make_monopole_harm(q: float, ell: float, m: float):
    r"""Create a monopole harmonic function :math:`Y_{q,\ell,m}`.

    Args:
        q: Monopole charge.
        ell: Angular momentum quantum number.
        m: Magnetic quantum number.

    Returns:
        A callable ``f(u, v) -> complex array`` evaluated in a local spinor frame.
    """
    norm_factor = np.sqrt(
        ((2 * ell + 1) / (4 * np.pi))
        * (ss.factorial(ell - m) * ss.factorial(ell + m))
        / (ss.factorial(ell - q) * ss.factorial(ell + q))
    )
    s_all = np.arange(int(ell - m) + 1)
    coeffs_all = (
        (-1) ** (ell - m - s_all)
        * ss.comb(ell - q, s_all)
        * ss.comb(ell + q, ell - m - s_all)
    )
    # Keep only terms with nonzero weight: zero-weight terms can carry
    # negative powers (0**-1 = inf), giving 0*inf = NaN at the poles.
    # Survivor exponents are all non-negative integers; each power below
    # uses a Python int so JAX dispatches to integer_pow, finite at 0.
    terms_spec = [
        (float(c), round(s + m + q), round(ell - s - m), int(s), round(ell - s - q))
        for s, c in zip(s_all, coeffs_all)
        if c != 0
    ]

    def Y_qlm(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the monopole harmonic from a local spinor frame.

        Returns:
            The monopole harmonic at every electron coordinate.
        """
        total = jnp.zeros_like(u)
        conj_u = jnp.conj(u)
        conj_v = jnp.conj(v)
        for coeff, a, b, c, d in terms_spec:
            total = total + coeff * (u**a) * (v**b) * (conj_u**c) * (conj_v**d)
        return norm_factor * total

    return Y_qlm


class Free(HallWavefunction):
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

    def _logpsi_from_spinor(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        orbitals = jnp.stack([orbital(u, v) for orbital in self.orbital_fns])
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)
        return jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax
