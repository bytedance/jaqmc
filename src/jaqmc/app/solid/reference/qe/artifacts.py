# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Read Quantum ESPRESSO output files from a ``.save`` directory.

XML metadata and HDF5 wavefunction files are combined into the typed models
used to build a solid reference.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

__all__ = [
    "BandChannel",
    "QECalculation",
    "WavefunctionBlock",
    "read_qe_calculation",
]


@dataclass(frozen=True)
class WavefunctionBlock:
    """One decoded QE wavefunction file labeled by its Cartesian k-point."""

    coefficients: np.ndarray
    miller_indices: np.ndarray
    kpoint: np.ndarray
    spin: int
    kpoint_index: int


@dataclass(frozen=True)
class BandChannel:
    """One spin channel with one band record for each k-point."""

    kpoints: np.ndarray
    occupations: tuple[np.ndarray, ...]
    eigenvalues: tuple[np.ndarray, ...]
    wavefunctions: tuple[WavefunctionBlock, ...]

    def __post_init__(self) -> None:
        nk = len(self.kpoints)
        if (
            len(self.occupations) != nk
            or len(self.eigenvalues) != nk
            or len(self.wavefunctions) != nk
        ):
            raise ValueError(
                "QE channel data must contain one entry per physical k-point."
            )
        if any(
            len(occupation) != len(energy)
            for occupation, energy in zip(
                self.occupations, self.eigenvalues, strict=True
            )
        ):
            raise ValueError("QE occupations and eigenvalues must have matching bands.")


@dataclass(frozen=True)
class QECalculation:
    """Collinear QE result with Cartesian k-points split by spin channel."""

    symbols: tuple[str, ...]
    atom_coords: np.ndarray
    lattice: np.ndarray
    alpha: BandChannel
    beta: BandChannel
    spin_polarized: bool


def read_qe_calculation(save_dir: Path) -> QECalculation:
    """Read QE XML and HDF5 output into a normalized calculation.

    Returns:
        The normalized QE calculation.
    """
    root = ET.parse(save_dir / "data-file-schema.xml").getroot()
    _require_unreduced_mesh(root)
    output = _require(root, "{*}output")
    _require_scf_converged(output)
    _require_pbe(output)
    symbols, positions, lattice, alat = _read_atomic_structure(output)
    kpoints, alpha_occ, alpha_energy, beta_occ, beta_energy, spin_polarized = (
        _normalized_band_channels(output, alat)
    )
    blocks = _read_wavefunction_blocks(save_dir, kpoints, spin_polarized)
    alpha = BandChannel(
        kpoints=kpoints,
        occupations=alpha_occ,
        eigenvalues=alpha_energy,
        wavefunctions=_match_wavefunctions(blocks, kpoints, spin=1),
    )
    beta = BandChannel(
        kpoints=kpoints,
        occupations=beta_occ,
        eigenvalues=beta_energy,
        wavefunctions=(
            _match_wavefunctions(blocks, kpoints, spin=2)
            if spin_polarized
            else alpha.wavefunctions
        ),
    )
    return QECalculation(symbols, positions, lattice, alpha, beta, spin_polarized)


# XML decoding ----------------------------------------------------------------


def _require(element: ET.Element, path: str) -> ET.Element:
    result = element.find(path)
    if result is None:
        raise ValueError(
            f"Quantum ESPRESSO XML is missing {path.rsplit('}', 1)[-1]!r}."
        )
    return result


def _numbers(element: ET.Element) -> np.ndarray:
    if element.text is None:
        raise ValueError(
            "Quantum ESPRESSO XML element "
            f"{element.tag.rsplit('}', 1)[-1]!r} has no data."
        )
    return np.fromstring(element.text, sep=" ")


def _qe_boolean(value: object) -> bool:
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).strip().lower() in {"true", "t", "1", ".true."}


def _require_unreduced_mesh(root: ET.Element) -> None:
    """Require saved QE input flags for an unreduced k-point mesh.

    Raises:
        ValueError: If QE did not record ``nosym=.true.`` and ``noinv=.true.``.
    """
    input_node = root.find("{*}input")
    if input_node is None:
        raise ValueError(
            "QE conversion requires saved input symmetry flags proving an unreduced "
            "k-point mesh; <input> is missing."
        )
    flags = input_node.find("{*}symmetry_flags")
    if flags is None:
        raise ValueError(
            "QE conversion requires nosym=.true. and noinv=.true.; "
            "<symmetry_flags> is missing."
        )
    for name in ("nosym", "noinv"):
        value = flags.find(f"{{*}}{name}")
        if value is None or not _qe_boolean(value.text):
            raise ValueError(
                "QE conversion requires an unreduced k-point mesh: "
                f"{name}=.true. was not recorded in data-file-schema.xml."
            )


def _require_scf_converged(output: ET.Element) -> None:
    convergence = output.find("{*}convergence_info")
    achieved = (
        convergence.find("{*}convergence_achieved") if convergence is not None else None
    )
    if achieved is not None and not _qe_boolean(achieved.text):
        raise ValueError("Quantum ESPRESSO SCF calculation did not converge.")


def _require_pbe(output: ET.Element) -> None:
    functional_node = output.find(".//{*}functional")
    functional = (
        functional_node.text.strip()
        if functional_node is not None and functional_node.text
        else None
    )
    if functional not in (None, "PBE", "pbe"):
        raise ValueError("Only PBE Quantum ESPRESSO references are supported.")


def _read_atomic_structure(
    output: ET.Element,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, float]:
    structure = _require(output, "{*}atomic_structure")
    alat = float(structure.attrib.get("alat", "1"))
    if alat <= 0:
        raise ValueError("QE XML alat must be positive.")
    cell = _require(structure, "{*}cell")
    lattice = np.stack(
        [_numbers(_require(cell, f"{{*}}{name}")) for name in ("a1", "a2", "a3")]
    )
    positions_node = _require(structure, "{*}atomic_positions")
    atoms = positions_node.findall("{*}atom")
    if not atoms:
        raise ValueError("QE XML atomic_structure has no atoms.")
    symbols = tuple(
        atom.attrib.get("name") or atom.attrib.get("species") or "" for atom in atoms
    )
    if not all(symbols):
        raise ValueError("QE XML atom is missing a name.")
    positions = np.stack([_numbers(atom) for atom in atoms])
    units = positions_node.attrib.get("units", "bohr").lower()
    if units == "alat":
        positions *= alat
    elif units == "crystal":
        positions = positions @ lattice
    elif units not in {"bohr", "a.u."}:
        raise ValueError(f"Unsupported QE atomic-position unit {units!r}.")
    return symbols, positions, lattice, alat


def _validate_electron_count(
    band_structure: ET.Element,
    entries: list[ET.Element],
    occupations: tuple[np.ndarray, ...],
) -> None:
    try:
        reported = float((_require(band_structure, "{*}nelec").text or "").strip())
        weights = [
            float(_require(entry, "{*}k_point").attrib["weight"]) for entry in entries
        ]
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "QE band structure must provide a total electron count and a weight "
            "for every k-point."
        ) from exc
    occupied = sum(
        weight * float(occupation.sum())
        for weight, occupation in zip(weights, occupations, strict=True)
    )
    if not np.isclose(occupied, reported):
        raise ValueError(
            "QE weighted occupations do not match its reported electron count: "
            f"{occupied} != {reported}."
        )


def _band_entries(
    output: ET.Element, alat: float
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    bool,
    ET.Element,
    list[ET.Element],
]:
    bands = _require(output, "{*}band_structure")
    noncolin = bands.find("{*}noncolin")
    if noncolin is not None and _qe_boolean(noncolin.text):
        raise ValueError("QE noncollinear/spinor references are not supported.")
    lsda = bands.find("{*}lsda")
    spin_polarized = lsda is not None and _qe_boolean(lsda.text)
    entries = bands.findall("{*}ks_energies")
    if not entries:
        raise ValueError("QE XML has no k-points.")
    kpoints = np.stack(
        [_numbers(_require(entry, "{*}k_point")) for entry in entries]
    ) * (2.0 * np.pi / alat)
    occupations = tuple(
        _numbers(_require(entry, "{*}occupations")) for entry in entries
    )
    eigenvalues = tuple(
        _numbers(_require(entry, "{*}eigenvalues")) for entry in entries
    )
    _validate_electron_count(bands, entries, occupations)
    return kpoints, occupations, eigenvalues, spin_polarized, bands, entries


def _normalized_band_channels(
    output: ET.Element, alat: float
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    bool,
]:
    """Split either QE spin-polarized band layout into alpha and beta meshes.

    Returns:
        Physical k-points, alpha occupations/energies, beta occupations/energies,
        and whether the calculation is spin-polarized.

    Raises:
        ValueError: If QE spin or band metadata is inconsistent.
    """
    kpoints, occupations, eigenvalues, spin_polarized, bands, entries = _band_entries(
        output, alat
    )
    if not spin_polarized:
        return kpoints, occupations, eigenvalues, occupations, eigenvalues, False

    nks = bands.find("{*}nks")
    if nks is None:
        packed = False
    else:
        try:
            packed = len(entries) == int((nks.text or "").strip())
        except ValueError as exc:
            raise ValueError(
                "QE spin-polarized band structure has an invalid nks."
            ) from exc
    if packed:
        try:
            nup = int((_require(bands, "{*}nbnd_up").text or "").strip())
            ndown = int((_require(bands, "{*}nbnd_dw").text or "").strip())
        except ValueError as exc:
            raise ValueError(
                "Packed-spin QE output must provide integer nbnd_up and nbnd_dw."
            ) from exc
        if (
            nup <= 0
            or ndown <= 0
            or any(
                len(values) != nup + ndown for values in (*occupations, *eigenvalues)
            )
        ):
            raise ValueError(
                "Packed-spin QE bands must contain positive nbnd_up + nbnd_dw values."
            )
        return (
            kpoints,
            tuple(values[:nup] for values in occupations),
            tuple(values[:nup] for values in eigenvalues),
            tuple(values[nup:] for values in occupations),
            tuple(values[nup:] for values in eigenvalues),
            True,
        )

    if len(kpoints) % 2:
        raise ValueError("Spin-polarized QE output must contain paired k-points.")
    split = len(kpoints) // 2
    if not np.allclose(kpoints[:split], kpoints[split:]):
        raise ValueError("QE spin channels must use the same k-point mesh.")
    return (
        kpoints[:split],
        occupations[:split],
        eigenvalues[:split],
        occupations[split:],
        eigenvalues[split:],
        True,
    )


# HDF5 wavefunction decoding --------------------------------------------------


def _read_wavefunction_blocks(
    save_dir: Path, kpoints: np.ndarray, spin_polarized: bool
) -> list[WavefunctionBlock]:
    files = sorted(save_dir.glob("*wfc*.hdf5"))
    if not files:
        raise ValueError(
            "QE conversion requires portable HDF5 wavefunction files (wfc*.hdf5)."
        )
    required_files = len(kpoints) * (2 if spin_polarized else 1)
    if len(files) != required_files:
        raise ValueError(
            "QE k-point metadata and portable wavefunction-file counts differ."
        )
    return [_decode_wavefunction_file(path) for path in files]


def _complex_coefficients(values: np.ndarray, *, igwx: int, nbnd: int) -> np.ndarray:
    values = np.asarray(values)
    expected_shape = (nbnd, 2 * igwx)
    if np.iscomplexobj(values) or values.shape != expected_shape:
        raise ValueError(
            "QE wavefunction dataset 'evc' must be a real array shaped "
            f"{expected_shape}, with interleaved real and imaginary coefficients."
        )
    return values[:, 0::2] + 1j * values[:, 1::2]


def _complete_plane_wave_expansion(
    miller_indices: np.ndarray, coefficients: np.ndarray, *, gamma_only: bool
) -> tuple[np.ndarray, np.ndarray]:
    if miller_indices.shape != (coefficients.shape[1], 3):
        raise ValueError("QE Miller indices do not match wavefunction coefficients.")
    if not gamma_only:
        return miller_indices, coefficients
    miller_set = {tuple(index) for index in miller_indices}
    if any(
        index != (0, 0, 0) and tuple(-value for value in index) in miller_set
        for index in miller_set
    ):
        raise ValueError(
            "Gamma-only QE wavefunction contains both G and -G coefficients."
        )
    nonzero = np.any(miller_indices != 0, axis=1)
    return (
        np.concatenate([miller_indices, -miller_indices[nonzero]]),
        np.concatenate([coefficients, np.conj(coefficients[:, nonzero])], axis=1),
    )


def _decode_wavefunction_file(path: Path) -> WavefunctionBlock:
    with h5py.File(path, "r") as handle:
        if "MillerIndices" not in handle or "evc" not in handle:
            raise ValueError(f"{path} is not a supported QE wavefunction HDF5 file.")
        required = {
            "gamma_only",
            "igwx",
            "ik",
            "ispin",
            "nbnd",
            "npol",
            "scale_factor",
            "xk",
        }
        missing = required.difference(handle.attrs)
        if missing:
            raise ValueError(
                f"{path} is missing QE attributes: {', '.join(sorted(missing))}."
            )
        igwx, nbnd, npol = (
            int(handle.attrs[name]) for name in ("igwx", "nbnd", "npol")
        )
        if npol != 1:
            raise ValueError(
                f"{path} contains npol={npol}; QE spinor wavefunctions are "
                "not supported."
            )
        if not np.isclose(float(handle.attrs["scale_factor"]), 1.0):
            raise ValueError(
                f"{path} has unsupported scale_factor={handle.attrs['scale_factor']}."
            )
        indices = np.asarray(handle["MillerIndices"], dtype=int)
        if indices.shape != (igwx, 3):
            raise ValueError(f"{path} MillerIndices must have shape ({igwx}, 3).")
        indices, coefficients = _complete_plane_wave_expansion(
            indices,
            _complex_coefficients(np.asarray(handle["evc"]), igwx=igwx, nbnd=nbnd),
            gamma_only=_qe_boolean(handle.attrs["gamma_only"]),
        )
        return WavefunctionBlock(
            coefficients=coefficients,
            miller_indices=indices,
            kpoint=np.asarray(handle.attrs["xk"], dtype=float),
            spin=int(handle.attrs["ispin"]),
            kpoint_index=int(handle.attrs["ik"]),
        )


def _match_wavefunctions(
    blocks: list[WavefunctionBlock],
    kpoints: np.ndarray,
    *,
    spin: int,
) -> tuple[WavefunctionBlock, ...]:
    """Return HDF5 blocks ordered by their QE k-point index.

    Returns:
        Wavefunction blocks ordered by the input k-point mesh.

    Raises:
        ValueError: If HDF5 spin labels or k-points do not match XML metadata.
    """
    mismatch = (
        "QE wavefunction-file k-points or spin labels do not match "
        "data-file-schema.xml."
    )
    channel = [block for block in blocks if block.spin == spin]
    by_index = {block.kpoint_index: block for block in channel}
    if len(channel) != len(kpoints) or len(by_index) != len(kpoints):
        raise ValueError(mismatch)
    try:
        result = tuple(by_index[index] for index in range(1, len(kpoints) + 1))
    except KeyError as exc:
        raise ValueError(mismatch) from exc
    if not all(
        np.allclose(block.kpoint, kpoint, atol=1e-8, rtol=0)
        for block, kpoint in zip(result, kpoints, strict=True)
    ):
        raise ValueError(mismatch)
    return result
