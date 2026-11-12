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
    def _eval_channel_orbitals(
        self, positions: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate occupied alpha and beta orbitals at flat electron positions."""

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
        alpha_flat, beta_flat = self._eval_channel_orbitals(flat)
        alpha = jnp.reshape(alpha_flat, (*leading, nelec, -1))
        beta = jnp.reshape(beta_flat, (*leading, nelec, -1))
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

    def _eval_channel_orbitals(
        self, positions: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        wrapped = wrap_positions(positions, jnp.asarray(self.lattice))
        displacement = positions - wrapped
        aos = self.evaluator(wrapped, jnp.asarray(self.kpoints))
        phases = jnp.exp(1j * jnp.einsum("ki,ni->kn", self.kpoints, displacement))
        basis_values = aos * phases[..., None]
        return (
            self._eval_mos(basis_values, self.alpha_coeffs, self.alpha_counts),
            self._eval_mos(basis_values, self.beta_coeffs, self.beta_counts),
        )

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
    """Occupied Bloch orbitals in separate plane-wave bases for each spin.

    Plane-wave Bloch orbitals are transformed onto a 2x-oversampled periodic
    3D cubic B-spline grid via inverse FFT at initialization and evaluated in
    real space by local 4x4x4 cubic B-spline interpolation.
    """

    alpha_g_vectors: np.ndarray
    beta_g_vectors: np.ndarray

    _inv_lattice: jnp.ndarray = field(init=False, repr=False, compare=False)
    _alpha_spline: jnp.ndarray = field(init=False, repr=False, compare=False)
    _beta_spline: jnp.ndarray = field(init=False, repr=False, compare=False)
    _alpha_kpoints: jnp.ndarray = field(init=False, repr=False, compare=False)
    _beta_kpoints: jnp.ndarray = field(init=False, repr=False, compare=False)

    _kind: ClassVar[str] = "plane_wave"
    _GRID_OVERSAMPLING: ClassVar[int] = 2
    _MIN_AXIS_GRID_SIZE: ClassVar[int] = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        alpha_miller = self._validate_and_extract_miller_indices(
            self.alpha_g_vectors,
            n_basis=self.alpha_coeffs.shape[0],
            what="Alpha plane-wave G-vectors",
        )
        beta_miller = self._validate_and_extract_miller_indices(
            self.beta_g_vectors,
            n_basis=self.beta_coeffs.shape[0],
            what="Beta plane-wave G-vectors",
        )
        object.__setattr__(
            self,
            "_inv_lattice",
            jnp.asarray(np.linalg.inv(self.lattice)),
        )
        alpha_spline = self._build_bspline_grid(
            self.alpha_coeffs, alpha_miller, self.alpha_counts
        )
        if (
            np.array_equal(self.alpha_counts, self.beta_counts)
            and np.array_equal(self.alpha_coeffs, self.beta_coeffs)
            and np.array_equal(alpha_miller, beta_miller)
        ):
            beta_spline = alpha_spline
        else:
            beta_spline = self._build_bspline_grid(
                self.beta_coeffs, beta_miller, self.beta_counts
            )
        object.__setattr__(self, "_alpha_spline", alpha_spline)
        object.__setattr__(self, "_beta_spline", beta_spline)
        kpoints = np.asarray(self.kpoints)
        indices = np.arange(len(kpoints))
        object.__setattr__(
            self,
            "_alpha_kpoints",
            jnp.asarray(kpoints[np.repeat(indices, self.alpha_counts)]),
        )
        object.__setattr__(
            self,
            "_beta_kpoints",
            jnp.asarray(kpoints[np.repeat(indices, self.beta_counts)]),
        )

    def _validate_and_extract_miller_indices(
        self, g_vectors: np.ndarray, *, n_basis: int, what: str
    ) -> np.ndarray:
        nk = len(self.kpoints)
        if g_vectors.ndim == 2:
            if g_vectors.shape != (n_basis, 3):
                raise ValueError(f"{what} must have shape ({n_basis}, 3).")
        elif g_vectors.ndim == 3:
            if g_vectors.shape != (nk, n_basis, 3):
                raise ValueError(f"{what} must have shape ({nk}, {n_basis}, 3).")
        else:
            raise ValueError(f"{what} must be rank-2 or rank-3.")
        scaled = np.asarray(g_vectors, dtype=np.float64) @ (
            np.asarray(self.lattice, dtype=np.float64).T / (2.0 * np.pi)
        )
        miller = np.rint(scaled)
        if not np.allclose(scaled, miller, atol=1e-5, rtol=0.0):
            raise ValueError(
                f"{what} must be integer combinations of the reciprocal lattice."
            )
        return miller.astype(np.int32)

    def _build_bspline_grid(
        self,
        coeffs: np.ndarray,
        miller: np.ndarray,
        counts: np.ndarray,
    ) -> jnp.ndarray:
        n_orb = int(coeffs.shape[1])
        if n_orb == 0 or coeffs.shape[0] == 0:
            return jnp.zeros((1, 1, 1, n_orb), dtype=complex)
        max_miller = np.max(np.abs(miller), axis=(0, 1) if miller.ndim == 3 else 0)
        grid_shape = tuple(
            max(
                self._MIN_AXIS_GRID_SIZE,
                self._GRID_OVERSAMPLING * (2 * int(m) + 1),
            )
            if int(m) > 0
            else 1
            for m in max_miller
        )
        grid_k = np.zeros((*grid_shape, n_orb), dtype=complex)
        start = 0
        for k_idx, count in enumerate(counts):
            c = int(count)
            if c == 0:
                continue
            stop = start + c
            m_k = miller[k_idx] if miller.ndim == 3 else miller
            idx = tuple(m_k[:, d] % grid_shape[d] for d in range(3))
            np.add.at(grid_k[..., start:stop], idx, coeffs[:, start:stop])
            start = stop
        prefilter_x = _bspline_prefilter_1d(grid_shape[0])[:, None, None, None]
        prefilter_y = _bspline_prefilter_1d(grid_shape[1])[None, :, None, None]
        prefilter_z = _bspline_prefilter_1d(grid_shape[2])[None, None, :, None]
        omega = float(abs(np.linalg.det(self.lattice)))
        scale = float(np.prod(grid_shape)) / np.sqrt(omega)
        grid_k_jax = jnp.asarray(grid_k) * (prefilter_x * prefilter_y * prefilter_z)
        return jnp.fft.ifftn(grid_k_jax, axes=(0, 1, 2)) * scale

    def _eval_channel_orbitals(
        self, positions: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        fractional = positions @ self._inv_lattice
        alpha = _eval_bspline_channel(
            self._alpha_spline, fractional, positions, self._alpha_kpoints
        )
        beta = _eval_bspline_channel(
            self._beta_spline, fractional, positions, self._beta_kpoints
        )
        return alpha, beta


def _bspline_prefilter_1d(size: int) -> jnp.ndarray:
    """Return 1D periodic cubic B-spline Fourier prefilter weights."""
    k = jnp.arange(size)
    return 3.0 / (2.0 + jnp.cos(2.0 * jnp.pi * k / size))


def _bspline_weights_1d(dt: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the 4 cubic B-spline basis functions for fractional offset ``dt``.

    Returns:
        Cubic B-spline basis weights with trailing dimension 4.
    """
    w_m1 = (1.0 - dt) ** 3 / 6.0
    w_0 = (3.0 * dt**3 - 6.0 * dt**2 + 4.0) / 6.0
    w_p1 = (-3.0 * dt**3 + 3.0 * dt**2 + 3.0 * dt + 1.0) / 6.0
    w_p2 = dt**3 / 6.0
    return jnp.stack([w_m1, w_0, w_p1, w_p2], axis=-1)


def _eval_bspline_channel(
    spline_coeffs: jnp.ndarray,
    fractional: jnp.ndarray,
    positions: jnp.ndarray,
    orbital_kpoints: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate periodic B-spline Bloch envelopes and apply Bloch phases.

    Returns:
        Occupied Bloch orbital values of shape ``(n_positions, n_orbitals)``.
    """
    if spline_coeffs.shape[-1] == 0:
        return jnp.zeros((positions.shape[0], 0), dtype=spline_coeffs.dtype)
    shape = jnp.asarray(spline_coeffs.shape[:3], dtype=fractional.dtype)
    scaled = (fractional - jnp.floor(fractional)) * shape
    i0 = jnp.floor(scaled).astype(jnp.int32)
    dt = scaled - i0

    wx = _bspline_weights_1d(dt[:, 0])
    wy = _bspline_weights_1d(dt[:, 1])
    wz = _bspline_weights_1d(dt[:, 2])

    offsets = jnp.asarray([-1, 0, 1, 2], dtype=jnp.int32)
    ix = (i0[:, 0, None] + offsets) % spline_coeffs.shape[0]
    iy = (i0[:, 1, None] + offsets) % spline_coeffs.shape[1]
    iz = (i0[:, 2, None] + offsets) % spline_coeffs.shape[2]

    patch = spline_coeffs[
        ix[:, :, None, None], iy[:, None, :, None], iz[:, None, None, :]
    ]
    envelope = jnp.einsum("na,nb,nc,nabcd->nd", wx, wy, wz, patch)
    phases = jnp.exp(1j * jnp.einsum("ni,mi->nm", positions, orbital_kpoints))
    return envelope * phases
