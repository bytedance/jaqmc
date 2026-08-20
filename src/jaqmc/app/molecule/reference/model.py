# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Persisted molecular orbital references."""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from jax import numpy as jnp
from upath import UPath

from jaqmc.utils.atomic.gto import AtomicOrbitalEvaluator
from jaqmc.utils.linalg import slogdet_blocks

__all__ = ["MoleculeReference"]


@dataclass(frozen=True)
class MoleculeReference:
    """Molecular orbitals represented in a spherical Gaussian atomic-orbital basis.

    Coefficient columns are occupied orbitals in the solver's occupied order.
    """

    symbols: tuple[str, ...]
    coords: np.ndarray
    charges: np.ndarray
    nspins: tuple[int, int]
    basis: dict[str, Any]
    alpha_coeffs: np.ndarray
    beta_coeffs: np.ndarray

    def __post_init__(self) -> None:
        if self.coords.shape != (len(self.symbols), 3):
            raise ValueError(
                "Molecular reference coordinates must have shape (natoms, 3)."
            )
        if self.charges.shape != (len(self.symbols),):
            raise ValueError("Molecular reference charges must have shape (natoms,).")
        if not np.issubdtype(self.charges.dtype, np.integer):
            raise ValueError("Molecular reference charges must have an integer dtype.")
        if len(self.nspins) != 2 or any(
            not isinstance(count, (int, np.integer)) or count < 0
            for count in self.nspins
        ):
            raise ValueError(
                "Molecular reference nspins must be two non-negative integers."
            )
        if self.alpha_coeffs.ndim != 2 or self.beta_coeffs.ndim != 2:
            raise ValueError("Molecular reference coefficients must be rank-2 arrays.")
        if self.alpha_coeffs.shape[0] != self.beta_coeffs.shape[0]:
            raise ValueError("Alpha and beta coefficients must use the same AO basis.")
        if self.alpha_coeffs.shape[1] != self.nspins[0]:
            raise ValueError("Alpha coefficients do not match alpha electron count.")
        if self.beta_coeffs.shape[1] != self.nspins[1]:
            raise ValueError("Beta coefficients do not match beta electron count.")

    @cached_property
    def evaluator(self) -> AtomicOrbitalEvaluator:
        """Spherical Gaussian AO evaluator for this reference."""
        atom_list = list(zip(self.symbols, self.coords.tolist(), strict=True))
        return AtomicOrbitalEvaluator(atom_list, self.basis)

    def eval_mos(self, positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate all stored alpha and beta molecular orbitals.

        Returns:
            Alpha and beta orbital values.
        """
        aos = self.evaluator(jnp.reshape(positions, (-1, 3)))
        alpha = aos @ jnp.asarray(self.alpha_coeffs)
        beta = aos @ jnp.asarray(self.beta_coeffs)
        leading = positions.shape[:-1]
        return (
            jnp.reshape(alpha, (*leading, alpha.shape[-1])),
            jnp.reshape(beta, (*leading, beta.shape[-1])),
        )

    def eval_orbitals(
        self, pos: jnp.ndarray, nspins: tuple[int, int]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate occupied orbital matrices for spin-separated electrons.

        Returns:
            Alpha and beta occupied orbital matrices.

        Raises:
            ValueError: If the requested spin counts differ from the reference.
        """
        n_alpha, n_beta = nspins
        if nspins != self.nspins:
            raise ValueError(
                f"Reference contains nspins={self.nspins}, but evaluation requested "
                f"{nspins}."
            )
        leading = pos.shape[:-2]
        flat = jnp.reshape(pos, (-1, 3))
        if n_alpha + n_beta == 0:
            empty = jnp.zeros((*leading, 0, 0))
            return empty, empty
        alpha, beta = self.eval_mos(flat)
        nelec = n_alpha + n_beta
        alpha = jnp.reshape(alpha, (*leading, nelec, -1))[..., :n_alpha, :]
        beta = jnp.reshape(beta, (*leading, nelec, -1))[..., n_alpha:, :]
        return alpha, beta

    def eval_slater(
        self, pos: jnp.ndarray, nspins: tuple[int, int] | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate the sign and log absolute determinant of the reference.

        Returns:
            The determinant sign and log absolute determinant.
        """
        return slogdet_blocks(*self.eval_orbitals(pos, nspins or self.nspins))

    def save(self, path: str | Path | UPath) -> None:
        """Save this reference in the canonical ``reference.npz`` format."""
        metadata = {
            "format": "jaqmc.molecule.reference",
            "version": 1,
            "symbols": list(self.symbols),
            "nspins": list(self.nspins),
            "basis": self.basis,
        }
        with UPath(path).open("wb") as output:
            np.savez(
                output,
                metadata=np.asarray(
                    json.dumps(metadata, default=lambda value: value.tolist())
                ),
                coords=self.coords,
                charges=self.charges,
                alpha_coeffs=self.alpha_coeffs,
                beta_coeffs=self.beta_coeffs,
            )

    @classmethod
    def load(cls, path: str | Path | UPath) -> "MoleculeReference":
        """Load a canonical molecular reference.

        Returns:
            The loaded reference.

        Raises:
            ValueError: If the file is not a supported molecular reference.
        """
        with (
            UPath(path).open("rb") as reference_file,
            np.load(reference_file, allow_pickle=False) as data,
        ):
            try:
                metadata = json.loads(str(data["metadata"].item()))
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid molecular reference file: {path}") from exc
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid molecular reference file: {path}.")
            if "format" not in metadata or "version" not in metadata:
                raise ValueError(f"Invalid molecular reference file: {path}.")
            if metadata.get("format") != "jaqmc.molecule.reference":
                raise ValueError(f"Unsupported molecular reference format in {path}.")
            if metadata.get("version") != 1:
                raise ValueError(f"Unsupported molecular reference version in {path}.")
            try:
                reference = cls(
                    symbols=tuple(metadata["symbols"]),
                    coords=data["coords"],
                    charges=data["charges"],
                    nspins=tuple(metadata["nspins"]),
                    basis=metadata["basis"],
                    alpha_coeffs=data["alpha_coeffs"],
                    beta_coeffs=data["beta_coeffs"],
                )
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
                raise ValueError(f"Invalid molecular reference file: {path}") from exc
        return reference
