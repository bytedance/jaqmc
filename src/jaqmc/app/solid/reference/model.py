# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Persisted solid-state orbital references."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from jax import numpy as jnp
from upath import UPath

from jaqmc.geometry.pbc import wrap_positions
from jaqmc.utils.atomic.gto import PBCAtomicOrbitalEvaluator
from jaqmc.utils.linalg import slogdet_blocks

__all__ = [
    "GaussianSolidReference",
    "PlaneWaveSolidReference",
    "SolidReference",
]


@dataclass(frozen=True)
class SolidReference(ABC):
    """Base class for solid references containing occupied Bloch orbitals.

    Coefficient columns are concatenated in solver k-point order and then
    ascending band order. The occupancy counts determine which k-point
    corresponds to each column.
    """

    symbols: tuple[str, ...]
    atom_coords: np.ndarray
    lattice: np.ndarray
    kpoints: np.ndarray
    alpha_counts: np.ndarray
    beta_counts: np.ndarray
    alpha_coeffs: np.ndarray
    beta_coeffs: np.ndarray

    _kind: ClassVar[str]

    def __post_init__(self) -> None:
        if self.atom_coords.shape != (len(self.symbols), 3):
            raise ValueError(
                "Solid reference atom coordinates must have shape (natoms, 3)."
            )
        if self.lattice.shape != (3, 3):
            raise ValueError("Solid reference lattice must have shape (3, 3).")
        if self.kpoints.ndim != 2 or self.kpoints.shape[1] != 3:
            raise ValueError("Solid reference k-points must have shape (nk, 3).")
        if (
            self.alpha_counts.shape != self.beta_counts.shape
            or self.alpha_counts.shape != (len(self.kpoints),)
        ):
            raise ValueError("Solid occupancy counts must have shape (nk,).")
        if np.any(self.alpha_counts < 0) or np.any(self.beta_counts < 0):
            raise ValueError("Solid occupancy counts must be non-negative.")
        if self.alpha_coeffs.ndim != 2 or self.beta_coeffs.ndim != 2:
            raise ValueError("Solid reference coefficients must be rank-2 arrays.")
        if self.alpha_coeffs.shape[1] != int(self.alpha_counts.sum()):
            raise ValueError("Alpha coefficients do not match alpha occupancy counts.")
        if self.beta_coeffs.shape[1] != int(self.beta_counts.sum()):
            raise ValueError("Beta coefficients do not match beta occupancy counts.")

    @property
    def nspins(self) -> tuple[int, int]:
        """Total occupied alpha and beta orbital counts."""
        return int(self.alpha_counts.sum()), int(self.beta_counts.sum())

    def get_orbital_kpoints(self) -> jnp.ndarray:
        """Return each orbital's k-point, with alpha orbitals before beta."""
        indices = np.arange(len(self.kpoints))
        alpha = np.repeat(indices, self.alpha_counts)
        beta = np.repeat(indices, self.beta_counts)
        return jnp.asarray(np.concatenate((self.kpoints[alpha], self.kpoints[beta])))

    def get_kpoint_occupancies(self) -> list[tuple[jnp.ndarray, int, int]]:
        """Return occupancies for every solver k-point, including empty ones."""
        return [
            (jnp.asarray(kpoint), int(alpha), int(beta))
            for kpoint, alpha, beta in zip(
                self.kpoints, self.alpha_counts, self.beta_counts, strict=True
            )
        ]

    @abstractmethod
    def _eval_basis_channels(
        self, positions: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate both spin-channel bases at a flat array of electron positions."""

    def _eval_mos(
        self,
        basis_values: jnp.ndarray,
        coeffs: np.ndarray,
        counts: np.ndarray,
    ) -> jnp.ndarray:
        pieces = []
        start = 0
        for kpoint, count in enumerate(counts):
            stop = start + int(count)
            pieces.append(basis_values[kpoint] @ jnp.asarray(coeffs[:, start:stop]))
            start = stop
        if not pieces:
            return jnp.zeros((basis_values.shape[1], 0), dtype=complex)
        return jnp.concatenate(pieces, axis=-1)

    def eval_orbitals(
        self, pos: jnp.ndarray, nspins: tuple[int, int] | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate occupied alpha and beta orbital matrices for a batch.

        Returns:
            Alpha and beta occupied orbital matrices.

        Raises:
            ValueError: If requested spin counts differ from the reference.
        """
        nspins = nspins or self.nspins
        if tuple(nspins) != self.nspins:
            raise ValueError(f"Reference contains nspins={self.nspins}, got {nspins}.")
        leading = pos.shape[:-2]
        nelec = sum(nspins)
        if nelec == 0:
            empty = jnp.zeros((*leading, 0, 0), dtype=complex)
            return empty, empty
        flat = jnp.reshape(pos, (-1, 3))
        alpha_basis, beta_basis = self._eval_basis_channels(flat)
        alpha = jnp.reshape(
            self._eval_mos(alpha_basis, self.alpha_coeffs, self.alpha_counts),
            (*leading, nelec, -1),
        )
        beta = jnp.reshape(
            self._eval_mos(beta_basis, self.beta_coeffs, self.beta_counts),
            (*leading, nelec, -1),
        )
        return (
            alpha[..., : nspins[0], : nspins[0]],
            beta[..., nspins[0] :, : nspins[1]],
        )

    def eval_slater(
        self, pos: jnp.ndarray, nspins: tuple[int, int] | None = None
    ) -> jnp.ndarray:
        """Evaluate the complex logarithm of the reference determinant.

        Returns:
            The complex determinant logarithm.
        """
        sign, logdet = slogdet_blocks(*self.eval_orbitals(pos, nspins))
        return logdet + jnp.log(sign)

    def _archive_contents(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Prepare the metadata and arrays persisted in a reference archive.

        Returns:
            The JSON metadata and NPZ array mapping.
        """
        values = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "symbols" and field.init
        }
        return (
            {
                "format": "jaqmc.solid.reference",
                "version": 1,
                "kind": self._kind,
                "symbols": list(self.symbols),
            },
            values,
        )

    def save(self, path: str | Path | UPath) -> None:
        """Save this reference in the canonical solid ``reference.npz`` format."""
        metadata, arrays = self._archive_contents()
        with UPath(path).open("wb") as output:
            np.savez(
                output,
                metadata=np.asarray(
                    json.dumps(metadata, default=lambda value: value.tolist())
                ),
                **cast(Any, arrays),
            )

    @classmethod
    def load(cls, path: str | Path | UPath) -> "SolidReference":
        """Load a canonical solid reference.

        Returns:
            A Gaussian or plane-wave reference.

        Raises:
            ValueError: If the file is not a supported solid reference.
        """
        with (
            UPath(path).open("rb") as reference_file,
            np.load(reference_file, allow_pickle=False) as data,
        ):
            try:
                metadata = json.loads(str(data["metadata"].item()))
                if not isinstance(metadata, dict):
                    raise ValueError
                if metadata.get("format") != "jaqmc.solid.reference":
                    raise ValueError
                if metadata.get("version") != 1:
                    raise ValueError
                symbols = tuple(metadata["symbols"])
                common = {
                    "symbols": symbols,
                    "atom_coords": data["atom_coords"],
                    "lattice": data["lattice"],
                    "kpoints": data["kpoints"],
                    "alpha_counts": data["alpha_counts"],
                    "beta_counts": data["beta_counts"],
                    "alpha_coeffs": data["alpha_coeffs"],
                    "beta_coeffs": data["beta_coeffs"],
                }
                kind = metadata["kind"]
                if kind == GaussianSolidReference._kind:
                    return GaussianSolidReference(
                        **common,
                        basis=metadata["basis"],
                        image_translation_vectors=data["image_translation_vectors"],
                    )
                if kind == PlaneWaveSolidReference._kind:
                    return PlaneWaveSolidReference(
                        **common,
                        alpha_g_vectors=data["alpha_g_vectors"],
                        beta_g_vectors=data["beta_g_vectors"],
                    )
                raise ValueError
            except (
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
                IndexError,
            ) as exc:
                raise ValueError(f"Invalid solid reference file: {path}") from exc


@dataclass(frozen=True)
class GaussianSolidReference(SolidReference):
    """Occupied Bloch orbitals in a periodic Gaussian atomic-orbital basis."""

    basis: dict[str, Any]
    image_translation_vectors: np.ndarray

    evaluator: PBCAtomicOrbitalEvaluator = field(init=False, repr=False, compare=False)
    _kind: ClassVar[str] = "gaussian"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.basis, dict):
            raise ValueError("Gaussian solid references require a Gaussian AO basis.")
        if self.image_translation_vectors.ndim != 2 or (
            self.image_translation_vectors.shape[1] != 3
        ):
            raise ValueError("Gaussian lattice images must have shape (nimages, 3).")
        object.__setattr__(
            self,
            "evaluator",
            PBCAtomicOrbitalEvaluator(
                list(zip(self.symbols, self.atom_coords.tolist(), strict=True)),
                self.basis,
                jnp.asarray(self.image_translation_vectors),
            ),
        )

    def _eval_basis_channels(
        self, positions: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        wrapped = wrap_positions(positions, jnp.asarray(self.lattice))
        displacement = positions - wrapped
        aos = self.evaluator(wrapped, jnp.asarray(self.kpoints))
        phases = jnp.exp(1j * jnp.einsum("ki,ni->kn", self.kpoints, displacement))
        basis_values = aos * phases[..., None]
        return basis_values, basis_values

    def _archive_contents(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Move the Gaussian basis definition from arrays into metadata.

        Returns:
            The JSON metadata and NPZ array mapping.
        """
        metadata, arrays = super()._archive_contents()
        metadata["basis"] = arrays.pop("basis")
        return metadata, arrays


@dataclass(frozen=True)
class PlaneWaveSolidReference(SolidReference):
    """Occupied Bloch orbitals in separate plane-wave bases for each spin."""

    alpha_g_vectors: np.ndarray
    beta_g_vectors: np.ndarray

    _kind: ClassVar[str] = "plane_wave"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_g_vectors(
            self.alpha_g_vectors,
            n_basis=self.alpha_coeffs.shape[0],
            what="Alpha plane-wave G-vectors",
        )
        self._validate_g_vectors(
            self.beta_g_vectors,
            n_basis=self.beta_coeffs.shape[0],
            what="Beta plane-wave G-vectors",
        )

    def _validate_g_vectors(
        self, g_vectors: np.ndarray, *, n_basis: int, what: str
    ) -> None:
        nk = len(self.kpoints)
        if g_vectors.ndim == 2:
            if g_vectors.shape != (n_basis, 3):
                raise ValueError(f"{what} must have shape ({n_basis}, 3).")
            return
        if g_vectors.ndim == 3:
            if g_vectors.shape != (nk, n_basis, 3):
                raise ValueError(f"{what} must have shape ({nk}, {n_basis}, 3).")
            return
        raise ValueError(f"{what} must be rank-2 or rank-3.")

    def _plane_wave_basis(
        self, positions: jnp.ndarray, g_vectors: np.ndarray
    ) -> jnp.ndarray:
        g_values = jnp.asarray(g_vectors)
        kpoints = jnp.asarray(self.kpoints)
        if g_values.ndim == 2:
            wavevectors = kpoints[:, None, :] + g_values[None, :, :]
        else:
            wavevectors = kpoints[:, None, :] + g_values
        phases = jnp.exp(1j * jnp.einsum("kgi,ni->kng", wavevectors, positions))
        return phases / jnp.sqrt(abs(np.linalg.det(self.lattice)))

    def _eval_basis_channels(
        self, positions: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return (
            self._plane_wave_basis(positions, self.alpha_g_vectors),
            self._plane_wave_basis(positions, self.beta_g_vectors),
        )
