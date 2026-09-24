# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from jaqmc.utils.atomic import PP_PH, Atom
from jaqmc.utils.atomic.pyscf import (
    PySCFPseudopotentials,
    automatic_double_zeta_basis,
    occupied_orbital_masks,
    resolve_pyscf_basis,
    translate_pp_to_pyscf,
)


def test_automatic_double_zeta_basis_maps_each_element() -> None:
    atoms = [
        Atom(symbol="H", coords=[0.0, 0.0, 0.0]),
        Atom(symbol="Li", coords=[0.0, 0.0, 1.0], pp="ccecp"),
        Atom(symbol="Fe", coords=[0.0, 0.0, 2.0], pp=PP_PH),
    ]

    assert automatic_double_zeta_basis(atoms) == {
        "H": "cc-pVDZ",
        "Li": "ccecpccpvdz",
        "Fe": "ccecpccpvdz",
    }


def test_automatic_double_zeta_basis_rejects_other_ecp_families() -> None:
    atom = Atom(symbol="Au", coords=[0.0, 0.0, 0.0], pp="lanl2dz")

    with pytest.raises(ValueError, match="does not define a basis"):
        automatic_double_zeta_basis([atom])


def test_resolve_pyscf_basis_overlays_mapping_on_automatic() -> None:
    atoms = [
        Atom(symbol="H", coords=[0.0, 0.0, 0.0]),
        Atom(symbol="Li", coords=[0.0, 0.0, 1.0], pp="ccecp"),
        Atom(symbol="Au", coords=[0.0, 0.0, 2.0], pp="lanl2dz"),
    ]

    assert resolve_pyscf_basis(atoms, {"Li": "sto-3g", "Au": "lanl2dz"}) == {
        "H": "cc-pVDZ",
        "Li": "sto-3g",
        "Au": "lanl2dz",
    }
    with pytest.raises(ValueError, match="does not define a basis"):
        resolve_pyscf_basis(atoms, {"H": "sto-3g"})
    with pytest.raises(ValueError, match="not used"):
        resolve_pyscf_basis(atoms[:1], {"Li": "sto-3g"})


@pytest.mark.parametrize(
    ("occupations", "unrestricted", "expected_alpha", "expected_beta"),
    [
        pytest.param(
            np.asarray([0.0, 1.0, 2.0]),
            False,
            np.asarray([False, True, True]),
            np.asarray([False, False, True]),
            id="restricted",
        ),
        pytest.param(
            np.asarray([[[0.0, 1.0]], [[1.0, 0.0]]]),
            True,
            np.asarray([[False, True]]),
            np.asarray([[True, False]]),
            id="unrestricted",
        ),
    ],
)
def test_occupied_orbital_masks(
    occupations, unrestricted, expected_alpha, expected_beta
):
    alpha, beta = occupied_orbital_masks(occupations, unrestricted=unrestricted)

    np.testing.assert_array_equal(alpha, expected_alpha)
    np.testing.assert_array_equal(beta, expected_beta)


@pytest.mark.parametrize(
    ("occupations", "unrestricted"),
    [
        pytest.param(np.asarray([1.5]), False, id="restricted"),
        pytest.param(np.asarray([[0.0], [0.5]]), True, id="unrestricted"),
    ],
)
def test_occupied_orbital_masks_reject_fractional_occupations(
    occupations, unrestricted
):
    with pytest.raises(ValueError, match="occupations are fractional"):
        occupied_orbital_masks(occupations, unrestricted=unrestricted)


@pytest.mark.parametrize(
    ("atom_specs", "solver_pp", "expected", "error"),
    [
        pytest.param(
            [("H", None)],
            {},
            PySCFPseudopotentials(ecp={}, pseudo={}),
            None,
            id="inherits-all-electron",
        ),
        pytest.param(
            [("Li", "ccecp")],
            {},
            PySCFPseudopotentials(ecp={"Li": "ccecp"}, pseudo={}),
            None,
            id="inherits-ecp",
        ),
        pytest.param(
            [("Fe", PP_PH)],
            {},
            PySCFPseudopotentials(ecp={"Fe": "ccecp"}, pseudo={}),
            None,
            id="inherits-ph",
        ),
        pytest.param(
            [("Li", "ccecp")],
            {"Li": "ccecp"},
            PySCFPseudopotentials(ecp={"Li": "ccecp"}, pseudo={}),
            None,
            id="overrides-with-ecp",
        ),
        pytest.param(
            [("Li", "ccecp"), ("F", "ccecp")],
            "ccecp",
            PySCFPseudopotentials(
                ecp={"Li": "ccecp", "F": "ccecp"},
                pseudo={},
            ),
            None,
            id="overrides-all-with-ecp",
        ),
        pytest.param(
            [("F", "ccecp")],
            {"F": "gth-pbe-q7"},
            PySCFPseudopotentials(ecp={}, pseudo={"F": "gth-pbe-q7"}),
            None,
            id="overrides-with-gth",
        ),
        pytest.param(
            [("Li", "ccecp")],
            {"Li": "gth-pbe-q3"},
            None,
            "must match the valence count",
            id="rejects-mismatched-valence",
        ),
        pytest.param(
            [("F", "ccecp")],
            {"F": "ccecp", "Li": "ccecp"},
            None,
            "not used",
            id="rejects-unused-override",
        ),
    ],
)
def test_translate_pp_to_pyscf(atom_specs, solver_pp, expected, error):
    atoms = [
        Atom(symbol=symbol, coords=[0.0, 0.0, 0.0], pp=pp) for symbol, pp in atom_specs
    ]

    if error is not None:
        with pytest.raises(ValueError, match=error):
            translate_pp_to_pyscf(atoms, solver_pp)
    else:
        assert translate_pp_to_pyscf(atoms, solver_pp) == expected
