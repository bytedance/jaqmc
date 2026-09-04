# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "runs" / "he_lit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate He LIT tables, report, and figures.",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RUN_ROOT / "lit" / "lit_spectrum.npz",
    )
    parser.add_argument(
        "--fit-path",
        type=Path,
        default=DEFAULT_RUN_ROOT / "post" / "fit.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "post",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "post" / "fig",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


args = parse_args()
RAW_PATH = args.raw_path
FIT_PATH = args.fit_path
OUT = args.output_dir
FIG = args.figure_dir

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_WAVENUMBER = 219474.63136320

# NIST He I reference data. Energies are relative to the 1s^2 1S ground state.
# A values and wavelengths are used only for validation, never to initialize
# or constrain the JaQMC line fit.
NIST_LINES = (
    ("1s2p 1P1", 171134.8970, 584.33436, 17.989),
    ("1s3p 1P1", 186209.3651, 537.02992, 5.6634),
    ("1s4p 1P1", 191492.7120, 522.21309, 2.4356),
)


def jackknife_standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    squared_deviation_sum = np.sum((values - mean) ** 2)
    return float(np.sqrt((values.size - 1) / values.size * squared_deviation_sum))


def nist_oscillator_strength(wavelength_angstrom: float, a_ki_1e8: float) -> float:
    # f_ik = 1.4991938e-16 (g_k/g_i) lambda^2 A_ki, with g_k/g_i = 3
    # for 1S0 -> 1P1 and A_ki in s^-1.
    return 1.4991938e-16 * 3.0 * wavelength_angstrom**2 * a_ki_1e8 * 1.0e8


OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


with np.load(RAW_PATH) as raw, np.load(FIT_PATH) as fit:
    omega = np.asarray(raw["omega"], dtype=np.float64)
    signed_lit = np.asarray(raw["signed_lit"], dtype=np.float64)[0]
    broadened = np.asarray(raw["broadened"], dtype=np.float64)[0]
    fidelity = np.asarray(raw["fidelity"], dtype=np.float64)[0]
    ess_fraction = np.asarray(raw["reweight_ess_fraction"], dtype=np.float64)[0]
    standard_error = np.asarray(raw["signed_lit_standard_error"], dtype=np.float64)[0]
    eta = float(np.asarray(raw["eta"]).item())

    fit_mask = np.asarray(fit["fit_mask"], dtype=np.bool_)
    fitted_lit_window = np.asarray(fit["fitted_lit"], dtype=np.float64)[0]
    fitted_lit = np.full_like(omega, np.nan)
    fitted_lit[fit_mask] = fitted_lit_window
    energy = float(np.asarray(fit["pole_energies"])[0])
    energy_ev = energy * HARTREE_TO_EV
    i_z = float(np.asarray(fit["pole_strengths"])[0, 0])
    oscillator_strength = 2.0 * energy * i_z

    energy_jk_se = float(np.asarray(fit["jackknife_pole_energy_standard_error"])[0])
    i_z_jk_se = float(np.asarray(fit["jackknife_pole_strength_standard_error"])[0, 0])
    loo_energy = np.asarray(fit["jackknife_leave_one_out_pole_energies"])[..., 0]
    loo_i_z = np.asarray(fit["jackknife_leave_one_out_pole_strengths"])[..., 0, 0]
    oscillator_strength_jk_se = jackknife_standard_error(2.0 * loo_energy * loo_i_z)

    raw_csv = OUT / "spectrum.csv"
    with raw_csv.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "omega_ha",
                "omega_ev",
                "signed_lit_z",
                "broadened_eta_over_pi_lit_z",
                "signed_lit_standard_error_z",
                "fidelity_z",
                "reweight_ess_fraction_z",
                "article_fit_k1_b0_z",
            )
        )
        for index in range(omega.size):
            writer.writerow(
                (
                    f"{omega[index]:.12f}",
                    f"{omega[index] * HARTREE_TO_EV:.12f}",
                    f"{signed_lit[index]:.12e}",
                    f"{broadened[index]:.12e}",
                    f"{standard_error[index]:.12e}",
                    f"{fidelity[index]:.12e}",
                    f"{ess_fraction[index]:.12e}",
                    f"{fitted_lit[index]:.12e}" if fit_mask[index] else "",
                )
            )

    nist = []
    for label, wavenumber, wavelength, a_value in NIST_LINES:
        reference_energy = wavenumber / HARTREE_TO_WAVENUMBER
        reference_f = nist_oscillator_strength(wavelength, a_value)
        nist.append((label, reference_energy, reference_f))

    properties_csv = OUT / "properties.csv"
    with properties_csv.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "transition",
                "status",
                "energy_ha",
                "energy_ev",
                "energy_jackknife_se_ha",
                "I_zn",
                "I_zn_jackknife_se",
                "oscillator_strength_isotropic",
                "oscillator_strength_jackknife_se",
                "nist_energy_ha",
                "nist_oscillator_strength_from_Aki",
            )
        )
        writer.writerow(
            (
                "1s^2 1S0 -> 1s2p 1P1",
                "reportable",
                f"{energy:.12f}",
                f"{energy_ev:.12f}",
                f"{energy_jk_se:.12e}",
                f"{i_z:.12f}",
                f"{i_z_jk_se:.12e}",
                f"{oscillator_strength:.12f}",
                f"{oscillator_strength_jk_se:.12e}",
                f"{nist[0][1]:.12f}",
                f"{nist[0][2]:.12f}",
            )
        )
        for label, reference_energy, reference_f in nist[1:]:
            writer.writerow(
                (
                    f"1s^2 1S0 -> {label}",
                    "not_reportable_from_this_spectrum",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    f"{reference_energy:.12f}",
                    f"{reference_f:.12f}",
                )
            )

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    ax.plot(omega, signed_lit, color="#155E75", linewidth=1.55, label="raw")
    ax.plot(
        omega[fit_mask],
        fitted_lit[fit_mask],
        color="#DC2626",
        linewidth=1.3,
        linestyle="--",
        label="fitted",
    )
    ax.axvline(energy, color="#7C3AED", linewidth=1.0, linestyle=":")
    ax.annotate(
        rf"$\omega_1={energy:.6f}$ Ha",
        xy=(energy, float(np.max(signed_lit))),
        xytext=(8, -25),
        textcoords="offset points",
        color="#5B21B6",
        ha="left",
        va="top",
    )
    ax.set_xlim(float(omega[0]), float(omega[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"Excitation energy $\omega$ (Ha)")
    ax.set_ylabel(r"$\mathcal{L}_z(\omega,\eta)$ (a.u.)")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")

    for suffix in ("png", "pdf"):
        fig.savefig(FIG / f"spectrum.{suffix}", dpi=300)
    plt.close(fig)

    summary = OUT / "report.md"
    summary.write_text(
        "\n".join(
            (
                "# He z-axis LIT report",
                "",
                "## Production output",
                "",
                f"- Raw spectrum: `{display_path(RAW_PATH)}`",
                f"- Grid: {omega.size} points, {omega[0]:.3f}-{omega[-1]:.3f} Ha",
                f"- eta: {eta:.3f} Ha",
                f"- fidelity range: {np.min(fidelity):.9f}-{np.max(fidelity):.9f}",
                f"- minimum ESS fraction: {np.min(ess_fraction):.9f}",
                "",
                "## Reportable transition",
                "",
                "Article single-width, ordinary unweighted least-squares fit with "
                "K=1, constant background, and the 0.750-0.830 Ha local window.",
                "",
                f"- excitation energy: {energy:.9f} Ha = {energy_ev:.6f} eV",
                f"- z-component transition strength I_zn: {i_z:.9f}",
                f"- isotropy-inferred oscillator strength f_0n = 2 omega_n I_zn: "
                f"{oscillator_strength:.9f}",
                f"- jackknife SE: energy {energy_jk_se:.3e} Ha; I_zn "
                f"{i_z_jk_se:.3e}; f {oscillator_strength_jk_se:.3e}",
                "",
                "The standard JaQMC output leaves oscillator_strengths unavailable "
                "because only z was computed. The reported f uses exact atomic "
                "isotropy (I_xn = I_yn = I_zn), not a silent zero-fill.",
                "",
                "## Higher lines",
                "",
                "Do not report higher excitation energies or oscillator strengths "
                "from this run. Model-order/background sweeps do not give stable "
                "higher line centers, and the raw curve contains no resolved 3p or "
                "4p peak at the NIST validation positions.",
            )
        )
        + "\n"
    )

print(f"energy_ha={energy:.12f}")
print(f"energy_ev={energy_ev:.12f}")
print(f"I_z={i_z:.12f}")
print(f"oscillator_strength_isotropic={oscillator_strength:.12f}")
print(f"oscillator_strength_jackknife_se={oscillator_strength_jk_se:.12e}")
