# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Rebuild Li et al. Fig. 2h and optionally overlay JaQMC results."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def correlation_error(energy: float, hartree_fock: float, bf_dmc: float) -> float:
    """Return the percentage correlation error used in Fig. 2h."""
    return 100.0 * (1.0 - (energy - hartree_fock) / (bf_dmc - hartree_fock))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("fig2h_reference.json"),
    )
    parser.add_argument(
        "--jaqmc-results",
        type=Path,
        help=(
            "Optional JSON with rows containing rs, energy_per_electron, and "
            "standard_error_per_electron."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("fig2h.png"))
    args = parser.parse_args()

    reference = load_json(args.reference)
    rows = reference["rows"]
    mapping = reference["figure_2h_method_mapping"]
    labels = list(mapping)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:gray"]
    x = np.arange(len(rows), dtype=float)
    width = 0.19

    figure, axis = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    for method_index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        field = mapping[label]
        errors = [
            np.nan
            if row[field] is None
            else correlation_error(row[field], row["hartree_fock"], row["bf_dmc"])
            for row in rows
        ]
        offset = (method_index - (len(labels) - 1) / 2) * width
        axis.bar(x + offset, errors, width, label=label, color=color, edgecolor="black")

    if args.jaqmc_results:
        jaqmc_rows = {row["rs"]: row for row in load_json(args.jaqmc_results)["rows"]}
        errors = []
        error_bars = []
        for row in rows:
            result = jaqmc_rows[row["rs"]]
            denominator = abs(row["bf_dmc"] - row["hartree_fock"])
            errors.append(
                correlation_error(
                    result["energy_per_electron"],
                    row["hartree_fock"],
                    row["bf_dmc"],
                )
            )
            error_bars.append(
                100.0 * result["standard_error_per_electron"] / denominator
            )
        axis.errorbar(
            x,
            errors,
            yerr=error_bars,
            fmt="D",
            color="black",
            capsize=3,
            label="JaQMC",
        )

    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axis.set_xticks(x, [str(row["rs"]).removesuffix(".0") for row in rows])
    axis.set_xlabel(r"$r_s$ / Bohr")
    axis.set_ylabel("Correlation error")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:g} %")
    axis.legend()
    axis.set_ylim(-2.2, 10.0)
    figure.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
