# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Protocol definition for molecular wavefunctions."""

from typing import Protocol, runtime_checkable

from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.array_types import Params
from jaqmc.wavefunction.base import WavefunctionLike

__all__ = ["MoleculeWavefunction"]


@runtime_checkable
class MoleculeWavefunction(WavefunctionLike, Protocol):
    """Protocol for molecular wavefunctions.

    Defines the minimal interface that workflow code depends on.

    Usage in workflow:
        - ``full_det`` and ``orbitals``: Used for pretraining (orbital matching loss)
        - ``logpsi``: Used for both pretraining (log amplitude) and VMC training
        - ``phase_logpsi``: Used for ECP calculations requiring wavefunction ratios
    """

    # Whether this wavefunction uses full determinant (vs spin-split).
    # Used by pretraining to determine how to format target orbitals.
    full_det: bool

    def logpsi(self, params: Params, data: MoleculeData) -> jnp.ndarray:
        """Evaluate log|psi| given parameters and data.

        Used by both pretraining (log amplitude function) and VMC training.

        Args:
            params: Wavefunction parameters from ``init_params``.
            data: MoleculeData containing electrons and atoms.

        Returns:
            Scalar log-amplitude of the wavefunction.
        """
        ...

    def phase_logpsi(
        self, params: Params, data: MoleculeData
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Returns (phase, log|psi|) given parameters and data.

        Phase is :math:`\\pm 1` for real wavefunctions, or a complex
        unit value for complex wavefunctions.

        Args:
            params: Wavefunction parameters from ``init_params``.
            data: MoleculeData containing electrons and atoms.
        """

    def orbitals(self, params: Params, data: MoleculeData) -> jnp.ndarray:
        """Evaluate orbitals given parameters and data.

        Used by pretraining to compute orbital matching loss against SCF orbitals.

        Args:
            params: Wavefunction parameters from ``init_params``.
            data: MoleculeData containing electrons and atoms.

        Returns:
            Orbital matrix of shape ``(ndets, n_electrons, n_electrons)``.
        """
        ...
