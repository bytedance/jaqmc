# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Quantum ESPRESSO input preparation."""

from pathlib import Path

import numpy as np
from jax import numpy as jnp

from jaqmc.app.solid.config import SolidConfig, SolidQESolverConfig
from jaqmc.utils.supercell import (
    get_primitive_kpts_for_supercell,
    get_reciprocal_vectors,
)

__all__ = ["validate_upf_header"]


def _solid_kpoints(system: SolidConfig) -> tuple[np.ndarray, np.ndarray]:
    reciprocal_jax = get_reciprocal_vectors(jnp.asarray(system.lattice_vectors))
    kpoints = np.asarray(
        get_primitive_kpts_for_supercell(
            jnp.asarray(system.supercell_matrix),
            reciprocal_jax,
            jnp.asarray(system.twist),
        )
    )
    reciprocal = np.asarray(reciprocal_jax)
    return kpoints, kpoints @ np.linalg.inv(reciprocal)


def _parse_upf_valence(line: str) -> float | None:
    """Read a UPF v1 or v2 valence charge from a header line.

    Returns:
        The valence charge, or ``None`` when it cannot be parsed.
    """
    tokens = line.replace("=", " ").replace('"', " ").replace("'", " ").split()
    lowered = [token.lower().replace("_", "") for token in tokens]
    for index, token in enumerate(lowered):
        if token == "zvalence":
            try:
                return float(tokens[index + 1])
            except (IndexError, ValueError):
                return None
        if (
            token == "z"
            and index + 1 < len(lowered)
            and lowered[index + 1] == "valence"
        ):
            for candidate in (
                tokens[index - 1] if index else "",
                *tokens[index + 2 : index + 3],
            ):
                try:
                    return float(candidate)
                except ValueError:
                    pass
            return None
    return None


def validate_upf_header(path: str | Path, expected_valence: int) -> float:
    """Return the UPF valence charge if it matches the expected atomic valence.

    Returns:
        The validated valence charge.

    Raises:
        ValueError: If the charge is missing, invalid, or does not match.
    """
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "z_valence" not in line.lower() and "z valence" not in line.lower():
            continue
        valence = _parse_upf_valence(line)
        if valence is None:
            raise ValueError(f"Could not parse valence charge in UPF {path}.")
        if not np.isclose(valence, expected_valence):
            raise ValueError(
                f"UPF {path} declares valence charge {valence}, expected "
                f"{expected_valence} for the configured system."
            )
        return valence
    raise ValueError(f"UPF {path} has no recognizable z_valence marker.")


def _pseudo_name(symbol: str, pseudo_file: dict[str, str]) -> str:
    if symbol not in pseudo_file:
        raise ValueError(
            f"QE requires solver.pseudo_file.{symbol} to specify the UPF filename."
        )
    return pseudo_file[symbol]


def render_input(
    system: SolidConfig,
    solver: SolidQESolverConfig,
) -> str:
    """Render a complete Quantum ESPRESSO input for a PBE SCF calculation.

    Returns:
        The complete ``qe.in`` contents.

    Raises:
        ValueError: If the QE configuration or pseudopotentials are invalid.
    """
    if solver.pseudo_dir is None:
        raise ValueError("solver.pseudo_dir must specify a directory.")
    pseudo_dir = Path(solver.pseudo_dir).resolve()
    if not pseudo_dir.is_dir():
        raise ValueError(f"solver.pseudo_dir is not a directory: {pseudo_dir}.")
    if solver.smearing != "off" and solver.degauss <= 0:
        raise ValueError("solver.degauss must be positive when using smearing.")
    pseudo_file = solver.pseudo_file

    species = []
    for atom in system.atoms:
        filename = _pseudo_name(atom.symbol, pseudo_file)
        validate_upf_header(pseudo_dir / filename, atom.charge)
        species.append((atom.symbol, atom.atomic_number, filename))
    unique_species = list(dict.fromkeys(species))
    _, fractional_kpoints = _solid_kpoints(system)
    occupations = (
        [" occupations = 'fixed',"]
        if solver.smearing == "off"
        else [
            " occupations = 'smearing',",
            f" smearing = '{solver.smearing}',",
            f" degauss = {solver.degauss},",
        ]
    )
    lines = [
        "&control",
        " calculation = 'scf',",
        f" prefix = '{solver.prefix}',",
        f" pseudo_dir = '{pseudo_dir}',",
        " outdir = '.',",
        " wf_collect = .true.,",
        "/",
        "&system",
        f" ibrav = 0, nat = {len(system.atoms)}, ntyp = {len(unique_species)},",
        f" ecutwfc = {solver.pw_cutoff}, ecutrho = {solver.rho_cutoff},",
        f" tot_charge = {system.total_charge},",
        f" nspin = {2 if system.spin_imbalance else 1},",
        *(
            [f" tot_magnetization = {system.spin_imbalance},"]
            if system.spin_imbalance
            else []
        ),
        *occupations,
        " nosym = .true.,",
        " noinv = .true.,",
        "/",
        "&electrons",
        "/",
        "CELL_PARAMETERS bohr",
        *(" ".join(f"{value:.16g}" for value in row) for row in system.lattice_vectors),
        "ATOMIC_SPECIES",
        *(
            f"{symbol} {mass:.8g} {filename}"
            for symbol, mass, filename in unique_species
        ),
        "ATOMIC_POSITIONS bohr",
        *(
            f"{atom.symbol} " + " ".join(f"{value:.16g}" for value in atom.coords)
            for atom in system.atoms
        ),
        "K_POINTS crystal",
        str(len(fractional_kpoints)),
        *(
            " ".join(f"{value:.16g}" for value in (*kpoint, 1.0))
            for kpoint in fractional_kpoints
        ),
    ]
    return "\n".join(lines) + "\n"
