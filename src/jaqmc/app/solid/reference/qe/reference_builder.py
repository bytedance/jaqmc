# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Build plane-wave solid references from normalized QE bands."""

import logging
from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp

from jaqmc.app.solid.reference.model import PlaneWaveSolidReference
from jaqmc.utils.supercell import get_reciprocal_vectors

from .artifacts import BandChannel, QECalculation

__all__ = ["build_reference"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PackedSpinChannel:
    coefficients: np.ndarray
    orbital_counts: np.ndarray
    miller_indices: np.ndarray


def build_reference(calculation: QECalculation) -> PlaneWaveSolidReference:
    """Build JaQMC's canonical plane-wave reference from normalized QE bands.

    Returns:
        The converted plane-wave reference.
    """
    alpha, beta = _pack_spin_channels(calculation)
    return PlaneWaveSolidReference(
        symbols=calculation.symbols,
        atom_coords=calculation.atom_coords,
        lattice=calculation.lattice,
        kpoints=calculation.alpha.kpoints,
        alpha_counts=alpha.orbital_counts,
        beta_counts=beta.orbital_counts,
        alpha_coeffs=alpha.coefficients,
        beta_coeffs=beta.coefficients,
        alpha_g_vectors=_cartesian_g_vectors(alpha.miller_indices, calculation.lattice),
        beta_g_vectors=_cartesian_g_vectors(beta.miller_indices, calculation.lattice),
    )


def _pack_spin_channels(
    calculation: QECalculation,
) -> tuple[_PackedSpinChannel, _PackedSpinChannel]:
    if calculation.spin_polarized:
        counts = tuple(
            _rounded_electron_count(
                float(sum(occupation.sum() for occupation in channel.occupations)),
                "spin channel",
            )
            for channel in (calculation.alpha, calculation.beta)
        )
    else:
        # QE nspin=1 occupations are 0-2, so the per-spin QMC count is half
        # the occupation sum. Smearing does not change that scale.
        occupation_sum = float(
            sum(occupation.sum() for occupation in calculation.alpha.occupations)
        )
        count = _rounded_electron_count(occupation_sum / 2, "unpolarized spin channel")
        counts = (count, count)
    alpha = _pack_channel(
        calculation.alpha,
        _selected_bands(
            calculation.alpha,
            counts[0],
            spin_polarized=calculation.spin_polarized,
            spin=0,
        ),
    )
    beta = _pack_channel(
        calculation.beta,
        _selected_bands(
            calculation.beta,
            counts[1],
            spin_polarized=calculation.spin_polarized,
            spin=1,
        ),
    )
    return alpha, beta


def _fixed_occupations(
    occupations: tuple[np.ndarray, ...], allowed: tuple[float, ...]
) -> bool:
    return all(
        np.all(np.any(np.isclose(values[:, None], allowed, atol=1e-8), axis=1))
        for values in occupations
    )


def _rounded_electron_count(value: float, description: str) -> int:
    rounded = int(np.rint(value))
    if not np.isclose(value, rounded, atol=1e-8):
        raise ValueError(f"QE {description} electron count is not an integer: {value}.")
    return rounded


def _selected_bands(
    channel: BandChannel,
    electron_count: int,
    *,
    spin_polarized: bool,
    spin: int,
) -> tuple[np.ndarray, ...]:
    allowed = (0.0, 1.0) if spin_polarized else (0.0, 2.0)
    if _fixed_occupations(channel.occupations, allowed):
        return tuple(
            np.flatnonzero(occupation > 0.9) for occupation in channel.occupations
        )
    logger.warning(
        "QE fractional occupations were converted to a single QMC determinant "
        "by filling the %d lowest-energy %s orbitals.",
        electron_count,
        "alpha" if spin == 0 else "beta",
    )
    candidates = sorted(
        (float(energy), kpoint, band)
        for kpoint, energies in enumerate(channel.eigenvalues)
        for band, energy in enumerate(energies)
    )
    if electron_count > len(candidates):
        raise ValueError(
            "QE has fewer bands than the integer number of occupied QMC orbitals."
        )
    selected: list[list[int]] = [[] for _ in channel.kpoints]
    for _, kpoint, band in candidates[:electron_count]:
        selected[kpoint].append(band)
    return tuple(np.asarray(bands, dtype=int) for bands in selected)


def _pack_channel(
    channel: BandChannel, bands: tuple[np.ndarray, ...]
) -> _PackedSpinChannel:
    max_plane_waves = max(
        block.coefficients.shape[1] for block in channel.wavefunctions
    )
    counts = np.asarray([len(indices) for indices in bands], dtype=np.int32)
    dtype = np.result_type(
        *(block.coefficients.dtype for block in channel.wavefunctions)
    )
    coefficients = np.zeros((max_plane_waves, int(counts.sum())), dtype=dtype)
    miller_indices = np.zeros(
        (len(channel.kpoints), max_plane_waves, 3), dtype=np.int32
    )
    offset = 0
    for index, (block, selected) in enumerate(
        zip(channel.wavefunctions, bands, strict=True)
    ):
        nplane_waves = block.coefficients.shape[1]
        next_offset = offset + len(selected)
        coefficients[:nplane_waves, offset:next_offset] = block.coefficients[selected].T
        miller_indices[index, :nplane_waves] = block.miller_indices
        offset = next_offset
    return _PackedSpinChannel(coefficients, counts, miller_indices)


def _cartesian_g_vectors(miller_indices: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return miller_indices.astype(float) @ np.asarray(
        get_reciprocal_vectors(jnp.asarray(lattice))
    )
