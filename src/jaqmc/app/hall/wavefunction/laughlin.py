# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exact Laughlin wavefunction on the Haldane sphere (for benchmarking)."""

from collections.abc import Callable

from jax import numpy as jnp

from jaqmc.utils.wiring import runtime_dep

from .base import HallWavefunction, lll_monopole_harmonics

__all__ = ["Laughlin"]


def _jastrow_factor(u: jnp.ndarray, v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """CF-Jastrow pair elements and per-electron row products.

    Args:
        u: First spinor coordinate for every electron.
        v: Second spinor coordinate for every electron.

    Returns:
        ``(element, jastrow)`` where ``element[i, j] = u_i v_j - u_j v_i``
        with unit diagonal, and ``jastrow[i] = prod_j element[i, j]`` with
        shape ``(n_elec, 1)``.
    """
    element = u[:, None] * v[None, :] - u[None, :] * v[:, None] + jnp.eye(u.shape[0])
    return element, jnp.prod(element, axis=-1, keepdims=True)


class Laughlin(HallWavefunction):
    """Laughlin wavefunction for ground and quasiparticle/quasihole states.

    Constructs the Laughlin state as a Slater determinant of composite
    fermion orbitals with an attached Jastrow factor.

    Args:
        nspins: ``(n_up, n_down)`` electron counts.
        flux: Magnetic flux :math:`2Q`.
        flux_per_elec: Composite fermion flux attachment parameter :math:`p`.
        excitation_lz: Target :math:`L_z` for a quasiparticle or quasihole
            excitation. Ignored for ground-state fillings.
    """

    nspins: tuple[int, int] = runtime_dep()
    flux: int = runtime_dep()
    flux_per_elec: int = 1
    excitation_lz: int = 0

    def setup(self) -> None:
        nelec = sum(self.nspins)
        Q = self.flux / 2
        self.Q1 = Q - self.flux_per_elec * (nelec - 1)
        self._cf_orbitals: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        if nelec == 2 * self.Q1 + 1:
            self._cf_orbitals = self._full_orbitals
        elif nelec == 2 * self.Q1:
            self._check_lz()
            if not (-abs(self.Q1) <= self.excitation_lz <= abs(self.Q1)):
                raise ValueError(
                    f"excitation_lz={self.excitation_lz} out of range "
                    f"[-{abs(self.Q1)}, {abs(self.Q1)}] for quasihole."
                )
            self._cf_orbitals = self._quasihole_orbitals
        elif nelec == 2 * self.Q1 + 2:
            self._check_lz()
            if not (-abs(self.Q1) - 1 <= self.excitation_lz <= abs(self.Q1) + 1):
                raise ValueError(
                    f"excitation_lz={self.excitation_lz} out of range "
                    f"[-{abs(self.Q1) + 1}, {abs(self.Q1) + 1}] for quasiparticle."
                )
            self._cf_orbitals = self._quasiparticle_orbitals
        else:
            raise ValueError(
                f"Unsupported Laughlin filling: {nelec} electrons for "
                f"flux={self.flux} and flux_per_elec={self.flux_per_elec} "
                f"(Q1={self.Q1}). Expected nelec in "
                f"{{2*Q1, 2*Q1+1, 2*Q1+2}} for quasihole, ground, or "
                f"quasiparticle."
            )

    def _check_lz(self) -> None:
        """Validate that ``excitation_lz`` is compatible with ``Q1``.

        Raises:
            ValueError: If ``excitation_lz - Q1`` is not an integer.
        """
        diff = self.excitation_lz - self.Q1
        if int(diff) != diff:
            raise ValueError(
                f"Impossible excitation_lz={self.excitation_lz} for Q1={self.Q1}."
            )

    def _logpsi_from_spinor(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        orbitals = self._cf_orbitals(u, v)
        signs, logdets = jnp.linalg.slogdet(orbitals)
        logmax = jnp.max(logdets)
        return jnp.log(jnp.sum(signs * jnp.exp(logdets - logmax))) + logmax

    def _full_orbitals(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        harmonics = lll_monopole_harmonics(u, v, self.Q1)
        _, jastrow = _jastrow_factor(u, v)
        return harmonics * jastrow

    def _quasihole_orbitals(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Drop the m = -excitation_lz column (index Q - excitation_lz).
        # Column order affects only the determinant's global sign.
        full = lll_monopole_harmonics(u, v, self.Q1)
        excluded = round(self.Q1 - self.excitation_lz)
        harmonics = jnp.concatenate(
            [full[..., :excluded], full[..., excluded + 1 :]], axis=-1
        )
        _, jastrow = _jastrow_factor(u, v)
        return harmonics * jastrow

    def _quasiparticle_orbitals(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        Q = self.Q1
        harmonics = lll_monopole_harmonics(u, v, Q)

        element, jastrow = _jastrow_factor(u, v)
        jastrow_flat = jastrow[:, 0]
        d_log_jastrow_dv = jnp.sum(-u / element, axis=-1) + u
        d_log_jastrow_du = jnp.sum(v / element, axis=-1) - v

        m1 = self.excitation_lz
        a1 = round(Q + m1)
        b1 = round(Q - m1)
        prefactor = (u**a1) * (v**b1)
        excited = (
            prefactor
            * jastrow_flat
            * (
                (Q + 1 + m1) * v * d_log_jastrow_dv
                - (Q + 1 - m1) * u * d_log_jastrow_du
            )
        )[:, None]
        return jnp.concatenate([harmonics * jastrow, excited], axis=-1)
