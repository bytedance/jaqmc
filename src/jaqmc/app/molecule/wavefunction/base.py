# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Protocol and Flax execution adapters for molecular wavefunctions."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from flax import linen as nn
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.array_types import Params
from jaqmc.wavefunction.base import Wavefunction, WavefunctionLike
from jaqmc.wavefunction.output.logdet import RealLogDetOutput

__all__ = ["MoleculeWavefunction", "MoleculeWavefunctionModule"]


@runtime_checkable
class MoleculeWavefunction(WavefunctionLike[MoleculeData, RealLogDetOutput], Protocol):
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

    def logpsi(
        self, params: Params, data: MoleculeData, *, precision: str | None = None
    ) -> jnp.ndarray:
        """Evaluate log|psi| given parameters and data.

        Used by both pretraining (log amplitude function) and VMC training.

        Args:
            params: Wavefunction parameters from ``init_params``.
            data: MoleculeData containing electrons and atoms.
            precision: Evaluation mode. ``None`` inherits the configured default.

        Returns:
            Scalar log-amplitude of the wavefunction.
        """
        ...

    def phase_logpsi(
        self, params: Params, data: MoleculeData, *, precision: str | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Returns (phase, log|psi|) given parameters and data.

        Phase is :math:`\\pm 1` for real wavefunctions, or a complex
        unit value for complex wavefunctions.

        Args:
            params: Wavefunction parameters from ``init_params``.
            data: MoleculeData containing electrons and atoms.
            precision: Evaluation mode. ``None`` inherits the configured default.
        """

    def orbitals(
        self, params: Params, data: MoleculeData, *, precision: str | None = None
    ) -> jnp.ndarray:
        """Evaluate orbitals given parameters and data.

        Used by pretraining to compute orbital matching loss against SCF orbitals.

        Args:
            params: Wavefunction parameters from ``init_params``.
            data: MoleculeData containing electrons and atoms.
            precision: Evaluation mode. ``None`` inherits the configured default.

        Returns:
            Orbital matrix of shape ``(ndets, n_electrons, n_electrons)``.
        """
        ...


class MoleculeWavefunctionModule(Wavefunction[MoleculeData, RealLogDetOutput]):
    """Optional Flax adapters for real molecular wavefunctions.

    Concrete ansatzes define their physical computation in ``__call__`` and
    ``get_orbitals``. Evaluation overrides select immutable Flax modules with the
    same parameter layout, leaving the configured default unchanged.
    """

    @abstractmethod
    def get_orbitals(self, data: MoleculeData) -> jnp.ndarray:
        """Compute orbital matrices from electron and atom positions."""

    @nn.nowrap
    def evaluate(
        self, params: Params, data: MoleculeData, *, precision: str | None = None
    ) -> RealLogDetOutput:
        """Return structured wavefunction output with an optional mode."""
        module = self if precision is None else self.clone(precision=precision)
        return module.apply(params, data)  # type: ignore

    @nn.nowrap
    def logpsi(
        self, params: Params, data: MoleculeData, *, precision: str | None = None
    ) -> jnp.ndarray:
        """Return log-amplitude with explicit parameters and an optional mode."""
        return self.evaluate(params, data, precision=precision)["logpsi"]

    @nn.nowrap
    def phase_logpsi(
        self, params: Params, data: MoleculeData, *, precision: str | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the real sign and log-amplitude with an optional mode."""
        output = self.evaluate(params, data, precision=precision)
        return output["sign_logpsi"], output["logpsi"]

    @nn.nowrap
    def orbitals(
        self, params: Params, data: MoleculeData, *, precision: str | None = None
    ) -> jnp.ndarray:
        """Return orbital matrices with explicit parameters and an optional mode."""
        module = self if precision is None else self.clone(precision=precision)
        return module.apply(params, data, method=module.get_orbitals)  # type: ignore
