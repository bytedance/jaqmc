# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Reblock the six HEG evaluations used to reproduce Li et al. Fig. 2h."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyblock.blocking

N_ELECTRONS = 54
EXPECTED_RS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def correlation_error(energy: float, hartree_fock: float, bf_dmc: float) -> float:
    """Return the Fig. 2h correlation-energy error in percent."""
    return 100.0 * (1.0 - (energy - hartree_fock) / (bf_dmc - hartree_fock))


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reblock_energy(values: np.ndarray) -> dict[str, float | int]:
    """Return the pyblock optimum for one finite real-valued energy series.

    Returns:
        Optimal block metadata, mean, standard error, and error of that error.

    Raises:
        ValueError: If the input shape or values are invalid.
        RuntimeError: If the data do not contain an optimal reblocking plateau.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Energy series must be one-dimensional with at least 2 rows")
    if not np.isfinite(values).all():
        raise ValueError("Energy series contains non-finite values")

    blocks = pyblock.blocking.reblock(values)
    optimum = float(pyblock.blocking.find_optimal_block(values.size, blocks)[0])
    if not math.isfinite(optimum):
        raise RuntimeError(
            "Reblocking did not find an optimal plateau; collect more evaluation data"
        )
    result = blocks[int(optimum)]
    return {
        "level": int(result.block),
        "block_size_steps": 2 ** int(result.block),
        "remaining_blocks": int(result.ndata),
        "mean": float(np.asarray(result.mean).item()),
        "standard_error": float(np.asarray(result.std_err).item()),
        "standard_error_error": float(np.asarray(result.std_err_err).item()),
    }


def analyze_evaluation(
    path: Path,
    reference: dict[str, Any],
    *,
    expected_steps: int = 50_000,
) -> dict[str, Any]:
    """Analyze one fixed-wavefunction evaluation against its paper row.

    Returns:
        Reblocked JaQMC values, uncertainties, provenance, and paper differences.

    Raises:
        KeyError: If the required energy dataset is missing.
        ValueError: If the evaluation row count is not the required count.
    """
    with h5py.File(path, "r") as handle:
        if "total_energy" not in handle:
            raise KeyError(f"Missing total_energy in {path}")
        total_energy = np.asarray(handle["total_energy"][:])

    if total_energy.shape != (expected_steps,):
        raise ValueError(
            f"Expected {expected_steps} evaluation rows in {path}, "
            f"got shape {total_energy.shape}"
        )
    if not np.isfinite(total_energy).all():
        raise ValueError(f"Energy series contains non-finite values in {path}")
    energy_per_electron = total_energy / N_ELECTRONS
    blocked = reblock_energy(energy_per_electron.real)
    energy = float(blocked["mean"])
    standard_error = float(blocked["standard_error"])
    denominator = abs(reference["bf_dmc"] - reference["hartree_fock"])

    return {
        "rs": float(reference["rs"]),
        "evaluation_stats_h5": str(path.resolve()),
        "evaluation_stats_sha256": sha256(path),
        "evaluation_steps": expected_steps,
        "energy_per_electron": energy,
        "standard_error_per_electron": standard_error,
        "standard_error_error_per_electron": float(blocked["standard_error_error"]),
        "imaginary_energy_mean_per_electron": float(np.mean(energy_per_electron.imag)),
        "imaginary_energy_max_abs_per_electron": float(
            np.max(np.abs(energy_per_electron.imag))
        ),
        "reblocking": {
            key: value
            for key, value in blocked.items()
            if key not in {"mean", "standard_error", "standard_error_error"}
        },
        "correlation_error_percent": correlation_error(
            energy, reference["hartree_fock"], reference["bf_dmc"]
        ),
        "correlation_error_standard_error_percent": (
            100.0 * standard_error / denominator
        ),
        "paper_net_correlation_error_percent": correlation_error(
            reference["net"], reference["hartree_fock"], reference["bf_dmc"]
        ),
        "energy_minus_paper_net_hartree_per_electron": energy - reference["net"],
        "paper": {
            "hartree_fock": reference["hartree_fock"],
            "bf_dmc": reference["bf_dmc"],
            "net": reference["net"],
            "bf_vmc": reference["bf_vmc"],
            "dcd_figure_2h_tc_dcd": reference["tc_dcd"],
            "dcd_supplement_table_14": reference["dcd_supplement_table_14"],
            "tc_fciqmc": reference["tc_fciqmc"],
        },
    }


def parse_evaluation(value: str) -> tuple[float, Path]:
    """Parse an ``RS=PATH`` command-line argument.

    Returns:
        The density and evaluation HDF5 path.

    Raises:
        argparse.ArgumentTypeError: If the value is not an ``RS=PATH`` pair.
    """
    try:
        rs_text, path_text = value.split("=", maxsplit=1)
        return float(rs_text), Path(path_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected RS=PATH") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evaluation",
        action="append",
        type=parse_evaluation,
        required=True,
        metavar="RS=PATH",
        help="Repeat exactly once for each of r_s=0.5,1,2,5,10,20.",
    )
    parser.add_argument("--expected-steps", type=int, default=50_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation_paths = dict(args.evaluation)
    if len(evaluation_paths) != len(args.evaluation):
        raise ValueError("Each r_s may be specified only once")
    if set(evaluation_paths) != set(EXPECTED_RS):
        raise ValueError(
            f"Expected exactly r_s={EXPECTED_RS}, got {tuple(evaluation_paths)}"
        )

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    reference_rows = {float(row["rs"]): row for row in reference["rows"]}
    rows = [
        analyze_evaluation(
            evaluation_paths[rs],
            reference_rows[rs],
            expected_steps=args.expected_steps,
        )
        for rs in EXPECTED_RS
    ]
    report = {
        "contract": {
            "n_electrons": N_ELECTRONS,
            "energy_unit": "hartree per electron",
            "correlation_error_formula": (
                "100 * (1 - (energy - hartree_fock) / (bf_dmc - hartree_fock))"
            ),
            "figure_2h_dcd_mapping": "DCD label uses tc_dcd",
            "reblocking": "pyblock optimal block from evaluation_stats.h5",
        },
        "reference": str(args.reference.resolve()),
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
