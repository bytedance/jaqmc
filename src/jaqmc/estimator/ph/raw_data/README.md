# Pseudo-Hamiltonian (PH) reference XML tables

This directory bundles the PH radial tables consumed by
`jaqmc.estimator.ph.PHEnergy`. The files are tabulated in the
`r * V` convention used by the QMCPACK-style PH XML format. Each
file contains the angular-momentum channels needed to reconstruct
both the local PH potential `tilde_v_loc` and the `L^2` channel
`v_L2`; that reconstruction is implemented in
`jaqmc.estimator.ph.data._reconstruct_paper_ph_data`.

## Files and supported elements

The supported set tracks `jaqmc.estimator.ph.SUPPORTED_PH_ELEMENTS`
exactly. Each element below has a single corresponding XML file:

| Element | File          | Reference type   | Valence (Z) |
|---------|---------------|------------------|-------------|
| Cr      | `TM/Cr.cc.xml`| CCSD(T)-anchored | 14          |
| Mn      | `TM/Mn.hf.xml`| HF-anchored      | 15          |
| Fe      | `TM/Fe.cc.xml`| CCSD(T)-anchored | 16          |
| Co      | `TM/Co.cc.xml`| CCSD(T)-anchored | 17          |
| Ni      | `TM/Ni.hf.xml`| HF-anchored      | 18          |
| Cu      | `TM/Cu.hf.xml`| HF-anchored      | 19          |
| Zn      | `TM/Zn.cc.xml`| CCSD(T)-anchored | 20          |
| S       | `MG/S.cc.xml` | CCSD(T)-anchored | 6           |

The `cc` / `hf` filename suffix above is selected by the
`_PH_XML_INFO` table in `jaqmc.estimator.ph.data`, which also
routes each symbol to its `TM/` (transition metals) or `MG/`
(main group) subdirectory. The `Z` value is the PH effective
valence charge from `_PH_VALENCE_COUNTS` in
`jaqmc.utils.atomic.pp`; the two modules are kept separate so that
XML-loading details and the PH vocabulary do not depend on each
other (see the `jaqmc.estimator.ph.data` module docstring).

## Provenance

These XML files mirror the PH library distributed alongside the
original Bennett et al. (2022) PH paper and its Ichibha et al. (2023)
locality-error-free follow-up. They are vendored into this package so
that `jaqmc.estimator.ph` ships with self-contained reference data and
does not require any auxiliary package to be installed at runtime.

## Format

Each file is a QMCPACK-style PH XML document. The
`<semilocal l-local="...">` block lists `<vps l="s|p|d|...">` channels
in `r * V(r)` form. The PH paper truncated semilocal form
satisfies, for `l <= l_local`,

    <l|v_PH(M)|l> = tilde_v_loc + l(l+1) v_L2,

so `tilde_v_loc` and `v_L2` are recovered pointwise from the
tabulated channels.

## References

- Bennett, M. C., Reboredo, F. A., Mitas, L., Krogel, J. T.,
  "High Accuracy Transition Metal Effective Cores for the Many-Body
  Diffusion Monte Carlo Method,"
  *J. Chem. Theory Comput.* **18**, 828–839 (2022).
  [DOI: 10.1021/acs.jctc.1c00992](https://doi.org/10.1021/acs.jctc.1c00992)
- Ichibha, T., Nikaido, Y., Bennett, M. C., Krogel, J. T., Hongo, K.,
  Maezono, R., Reboredo, F. A.,
  "Locality error free effective core potentials for 3d transition
  metal elements developed for the diffusion Monte Carlo method,"
  *J. Chem. Phys.* **159**, 164114 (2023).
  [DOI: 10.1063/5.0175381](https://doi.org/10.1063/5.0175381)
- Fu, W. *et al.*, "Local Pseudopotential Unlocks the True Potential
  of Neural Network-based Quantum Monte Carlo,"
  [arXiv:2505.19909](https://arxiv.org/abs/2505.19909) (2025) —
  adds the sulfur PH parameterization and describes the
  NNQMC + PH + forward-Laplacian integration built on top of these
  tables.
