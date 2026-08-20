# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields

import numpy as np
import pytest
from upath import UPath

from jaqmc.app.solid.reference import PlaneWaveSolidReference, SolidReference


def _plane_wave_reference() -> PlaneWaveSolidReference:
    return PlaneWaveSolidReference(
        symbols=("He",),
        atom_coords=np.zeros((1, 3)),
        lattice=np.eye(3) * 4,
        # The second k-point has no occupied bands in either spin channel.
        kpoints=np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        alpha_counts=np.asarray([1, 0]),
        beta_counts=np.asarray([1, 0]),
        alpha_coeffs=np.ones((2, 1), dtype=complex),
        beta_coeffs=np.ones((2, 1), dtype=complex),
        alpha_g_vectors=np.zeros((2, 3)),
        beta_g_vectors=np.zeros((2, 3)),
    )


@pytest.mark.parametrize(
    "path_uri",
    ["{tmp_path}/reference.npz", "memory://jaqmc-solid-model/reference.npz"],
)
def test_plane_wave_reference_roundtrip(tmp_path, path_uri):
    reference = _plane_wave_reference()

    path = UPath(path_uri.format(tmp_path=tmp_path))
    reference.save(path)
    loaded = SolidReference.load(path)

    assert type(loaded) is type(reference)
    for reference_field in fields(reference):
        if not reference_field.compare:
            continue
        expected = getattr(reference, reference_field.name)
        actual = getattr(loaded, reference_field.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_load_rejects_reference_file_missing_kind_specific_arrays(tmp_path):
    complete_path = tmp_path / "complete.npz"
    _plane_wave_reference().save(complete_path)
    with np.load(complete_path) as data:
        arrays = dict(data.items())

    # A plane-wave archive without its G-vector arrays is incomplete.
    del arrays["alpha_g_vectors"]
    del arrays["beta_g_vectors"]
    incomplete_path = tmp_path / "incomplete.npz"
    np.savez(incomplete_path, **arrays)

    with pytest.raises(ValueError, match="Invalid solid reference file"):
        SolidReference.load(incomplete_path)
