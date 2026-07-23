# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exact Laughlin wavefunction on the Haldane sphere (for benchmarking)."""

from collections.abc import Callable

from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.array_types import Params
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import ComplexWFOutput, Wavefunction

__all__ = ["Laughlin"]


class Laughlin(Wavefunction[HallData, ComplexWFOutput]):
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

    def __call__(self, data: HallData) -> ComplexWFOutput:
        electrons = data.electrons
        theta, phi = electrons[..., 0], electrons[..., 1]
        u = (jnp.cos(theta / 2) * jnp.exp(0.5j * phi))[..., None]
        v = (jnp.sin(theta / 2) * jnp.exp(-0.5j * phi))[..., None]

        orbitals = self._cf_orbitals(u, v)
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

    def _quasihole_orbitals(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        Q = self.Q1
        m = jnp.concat(
            [
                jnp.arange(-Q, -self.excitation_lz),
                jnp.arange(Q, -self.excitation_lz, -1),
            ]
        )
        element = u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0])
        jastrow = jnp.prod(element, axis=-1, keepdims=True)
        return u ** (Q + m) * v ** (Q - m) * jastrow

    def _quasiparticle_orbitals(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        Q = self.Q1
        m = jnp.arange(-Q, Q + 1)
        orbitals = u ** (Q + m) * v ** (Q - m)

        element = u * v[:, 0] - u[:, 0] * v + jnp.eye(u.shape[0])
        jastrow = jnp.prod(element, axis=-1, keepdims=True)
        jastrow_dv = jastrow * (jnp.sum(-u[:, 0] / element, axis=-1, keepdims=True) + u)
        jastrow_du = jastrow * (jnp.sum(v[:, 0] / element, axis=-1, keepdims=True) - v)

        m1 = self.excitation_lz
        excited = (u ** (Q + m1) * v ** (Q - m1)) * (
            (Q + 1 + m1) * v * jastrow_dv - (Q + 1 - m1) * u * jastrow_du
        )
        return jnp.concat([orbitals * jastrow, excited], axis=-1)

    def logpsi(self, params: Params, data: HallData) -> jnp.ndarray:
        return self.evaluate(params, data)["logpsi"]
