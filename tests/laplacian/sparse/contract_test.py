# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Sparse metadata and constructor contracts for Forward Laplacian state."""

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.laplacian import (
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    OwnerRoles,
)
from tests.laplacian.helpers import assert_allclose


class TestOwnerRoleContract:
    def test_single_value_role_normalizes_to_constant(self):
        role = OwnerRole(2, np.array([4], dtype=np.int32))
        assert role.axis is None
        assert tuple(role.values.tolist()) == (4,)

    def test_reduce_output_axes_collapses_constant_owner_fiber(self):
        role = OwnerRole(0, np.array([2, 2], dtype=np.int32))
        reduced = role.reduce_output_axes((0,), output_ndim=2)
        assert reduced is not None
        assert reduced.axis is None
        assert tuple(reduced.values.tolist()) == (2,)

    def test_reduce_output_axes_rejects_mixed_owner_fiber(self):
        role = OwnerRole(0, np.array([0, 1], dtype=np.int32))
        assert role.reduce_output_axes((0,), output_ndim=2) is None

    def test_owner_roles_match_requires_exact_slot_layout(self):
        lhs = OwnerRoles(
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(None, np.array([3], dtype=np.int32)),
        )
        rhs = OwnerRoles(
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(None, np.array([3], dtype=np.int32)),
        )
        mismatch = OwnerRoles(
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(None, np.array([2], dtype=np.int32)),
        )

        assert lhs == rhs
        assert lhs != mismatch

    def test_owner_role_rejects_non_1d_values(self):
        with pytest.raises(ValueError, match="1D integer array"):
            OwnerRole(0, np.array([[0, 1]], dtype=np.int32))

    def test_owner_role_rejects_negative_axis(self):
        with pytest.raises(ValueError, match="non-negative"):
            OwnerRole(-1, np.array([0, 1], dtype=np.int32))

    def test_owner_role_rejects_multi_value_constant(self):
        with pytest.raises(ValueError, match="exactly one owner id"):
            OwnerRole(None, np.array([0, 1], dtype=np.int32))

    def test_owner_roles_reject_empty_input(self):
        with pytest.raises(ValueError, match="at least one OwnerRole"):
            OwnerRoles()

    def test_owner_roles_reject_non_owner_role_members(self):
        with pytest.raises(TypeError, match="explicit OwnerRole inputs"):
            OwnerRoles(
                OwnerRole(None, np.array([0], dtype=np.int32)),
                cast(Any, np.array([1], dtype=np.int32)),
            )


class TestSparseJacobianDenseContract:
    def test_local1_to_dense_maps_owner_ids_along_owner_axis(self):
        jacobian = Local1Jacobian(
            blocks=jnp.array(
                [[[[10.0], [20.0], [30.0]]]],
                dtype=jnp.float32,
            ),
            owners=OwnerRoles(OwnerRole(0, np.array([0, 1, 2], dtype=np.int32))),
            input_shape=(3, 1),
            input_owner_axis=0,
        )

        dense = jacobian.to_dense()

        assert dense.shape == (3, 3, 1)
        assert_allclose(
            dense,
            jnp.array(
                [
                    [[10.0], [0.0], [0.0]],
                    [[0.0], [20.0], [0.0]],
                    [[0.0], [0.0], [30.0]],
                ],
                dtype=jnp.float32,
            ),
        )

    def test_local2_to_dense_accumulates_matching_owner_slots(self):
        jacobian = Local2Jacobian(
            blocks=jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=jnp.float32),
            owners=OwnerRoles(
                OwnerRole(None, np.array([1], dtype=np.int32)),
                OwnerRole(None, np.array([1], dtype=np.int32)),
            ),
            input_shape=(3, 2),
            input_owner_axis=0,
        )

        dense = jacobian.to_dense()
        expected = jnp.array(
            [[0.0], [0.0], [4.0], [6.0], [0.0], [0.0]],
            dtype=jnp.float32,
        )
        assert dense.shape == (6, 1)
        assert_allclose(dense, expected)

    def test_local1_to_dense_preserves_nonzero_input_owner_axis_layout(self):
        jacobian = Local1Jacobian(
            blocks=jnp.array(
                [
                    [
                        [[10.0, 20.0], [30.0, 40.0]],
                        [[50.0, 60.0], [70.0, 80.0]],
                    ]
                ],
                dtype=jnp.float32,
            ),
            owners=OwnerRoles(OwnerRole(0, np.array([0, 1], dtype=np.int32))),
            input_shape=(2, 2),
            input_owner_axis=1,
        )

        dense = jacobian.to_dense()

        assert dense.shape == (4, 2, 2)
        assert_allclose(
            dense,
            jnp.array(
                [
                    [[10.0, 20.0], [0.0, 0.0]],
                    [[0.0, 0.0], [30.0, 40.0]],
                    [[50.0, 60.0], [0.0, 0.0]],
                    [[0.0, 0.0], [70.0, 80.0]],
                ],
                dtype=jnp.float32,
            ),
        )

    def test_local2_to_dense_preserves_nonzero_input_owner_axis_layout(self):
        jacobian = Local2Jacobian(
            blocks=jnp.array(
                [
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ],
                    [
                        [[10.0, 20.0], [30.0, 40.0]],
                        [[50.0, 60.0], [70.0, 80.0]],
                    ],
                ],
                dtype=jnp.float32,
            ),
            owners=OwnerRoles(
                OwnerRole(0, np.array([0, 1], dtype=np.int32)),
                OwnerRole(1, np.array([1, 0], dtype=np.int32)),
            ),
            input_shape=(2, 2),
            input_owner_axis=1,
        )

        dense = jacobian.to_dense()

        assert dense.shape == (4, 2, 2)
        assert_allclose(
            dense,
            jnp.array(
                [
                    [[1.0, 22.0], [0.0, 40.0]],
                    [[10.0, 0.0], [33.0, 4.0]],
                    [[5.0, 66.0], [0.0, 80.0]],
                    [[50.0, 0.0], [77.0, 8.0]],
                ],
                dtype=jnp.float32,
            ),
        )


class TestSparseJacobianConstruction:
    def test_local2_constructor_preserves_explicit_owner_roles(self):
        blocks = jnp.zeros((2, 1, 2, 4, 1), dtype=jnp.float32)
        jacobian = Local2Jacobian(
            blocks=blocks,
            owners=OwnerRoles(
                OwnerRole(0, np.arange(2, dtype=np.int32)),
                OwnerRole(1, np.arange(4, dtype=np.int32)),
            ),
            input_shape=(4, 1),
            input_owner_axis=0,
        )

        assert jacobian.blocks.shape == blocks.shape
        assert jacobian.owners[0].axis == 0
        assert jacobian.owners[1].axis == 1
        assert tuple(jacobian.owners[0].values.tolist()) == (0, 1)
        assert tuple(jacobian.owners[1].values.tolist()) == (0, 1, 2, 3)

    def test_sparse_constructor_rejects_empty_input_shape(self):
        with pytest.raises(ValueError, match="requires a non-empty input shape"):
            Local1Jacobian(
                blocks=jnp.zeros((1, 1), dtype=jnp.float32),
                owners=OwnerRoles(OwnerRole(None, np.array([0], dtype=np.int32))),
                input_shape=(),
                input_owner_axis=0,
            )

    def test_sparse_constructor_rejects_out_of_range_owner_axis(self):
        with pytest.raises(ValueError, match="input_owner_axis must be in"):
            Local1Jacobian(
                blocks=jnp.zeros((1, 1, 2), dtype=jnp.float32),
                owners=OwnerRoles(OwnerRole(0, np.array([0, 1], dtype=np.int32))),
                input_shape=(2, 1),
                input_owner_axis=2,
            )

    def test_sparse_constructor_rejects_missing_support_coord_axes(self):
        with pytest.raises(ValueError, match="fixed \\(support, coord\\) prefix"):
            Local1Jacobian(
                blocks=jnp.zeros((), dtype=jnp.float32),
                owners=OwnerRoles(OwnerRole(None, np.array([0], dtype=np.int32))),
                input_shape=(1, 1),
                input_owner_axis=0,
            )

    def test_sparse_constructor_rejects_support_coord_shape_mismatch(self):
        with pytest.raises(ValueError, match="support/coord mismatch"):
            Local1Jacobian(
                blocks=jnp.zeros((1, 1, 2, 2), dtype=jnp.float32),
                owners=OwnerRoles(OwnerRole(0, np.array([0, 1], dtype=np.int32))),
                input_shape=(2, 3),
                input_owner_axis=0,
            )

    def test_sparse_constructor_rejects_non_owner_roles_metadata(self):
        with pytest.raises(TypeError, match="requires explicit OwnerRoles"):
            Local1Jacobian(
                blocks=jnp.zeros((1, 1, 2), dtype=jnp.float32),
                owners=cast(
                    Any,
                    (OwnerRole(0, np.array([0, 1], dtype=np.int32)),),
                ),
                input_shape=(2, 1),
                input_owner_axis=0,
            )

    def test_sparse_constructor_rejects_wrong_owner_role_count(self):
        with pytest.raises(ValueError, match="requires exactly 1 owner roles"):
            Local1Jacobian(
                blocks=jnp.zeros((1, 1, 2), dtype=jnp.float32),
                owners=OwnerRoles(
                    OwnerRole(0, np.array([0, 1], dtype=np.int32)),
                    OwnerRole(None, np.array([0], dtype=np.int32)),
                ),
                input_shape=(2, 1),
                input_owner_axis=0,
            )

    def test_sparse_constructor_rejects_owner_ids_out_of_range(self):
        with pytest.raises(
            ValueError, match="owners must be in \\[0, input_n_particles\\)"
        ):
            Local1Jacobian(
                blocks=jnp.zeros((1, 1, 2), dtype=jnp.float32),
                owners=OwnerRoles(OwnerRole(0, np.array([0, 2], dtype=np.int32))),
                input_shape=(2, 1),
                input_owner_axis=0,
            )

    def test_sparse_constructor_rejects_unbroadcastable_owner_shape(self):
        with pytest.raises(ValueError, match="OwnerRoles shape axis 0 must be 1 or 2"):
            Local1Jacobian(
                blocks=jnp.zeros((1, 1, 2), dtype=jnp.float32),
                owners=OwnerRoles(OwnerRole(0, np.array([0, 1, 2], dtype=np.int32))),
                input_shape=(3, 1),
                input_owner_axis=0,
            )


class TestSparseJacobianShapeContract:
    def test_output_shape_matches_full_block_layout(self):
        jacobian = Local1Jacobian(
            blocks=jnp.arange(72.0, dtype=jnp.float32).reshape(1, 3, 4, 2, 3),
            owners=OwnerRoles(OwnerRole(1, np.array([0, 1], dtype=np.int32))),
            input_shape=(2, 3),
            input_owner_axis=0,
        )

        assert jacobian.output_shape == (4, 2, 3)
        assert jacobian.owners.factorized_shape(len(jacobian.output_shape)) == (1, 2, 1)
        assert jacobian.to_dense().shape == (6, 4, 2, 3)

    def test_with_blocks_derives_new_output_shape(self):
        jacobian = Local1Jacobian(
            blocks=jnp.arange(72.0, dtype=jnp.float32).reshape(1, 3, 4, 2, 3),
            owners=OwnerRoles(OwnerRole(1, np.array([0, 1], dtype=np.int32))),
            input_shape=(2, 3),
            input_owner_axis=0,
        )

        updated = jacobian.with_blocks(jacobian.blocks[:, :, :2])

        assert updated.output_shape == (2, 2, 3)
        assert updated.to_dense().shape == (6, 2, 2, 3)

    def test_with_blocks_rejects_owner_role_incompatible_with_new_shape(self):
        jacobian = Local1Jacobian(
            blocks=jnp.arange(72.0, dtype=jnp.float32).reshape(1, 3, 4, 2, 3),
            owners=OwnerRoles(OwnerRole(1, np.array([0, 1], dtype=np.int32))),
            input_shape=(2, 3),
            input_owner_axis=0,
        )

        with pytest.raises(ValueError) as exc_info:
            jacobian.with_blocks(jacobian.blocks[:, :, :, :1])
        message = str(exc_info.value)
        assert "OwnerRoles shape" in message
        assert "axis 1" in message

    def test_with_blocks_accepts_replacement_owners(self):
        jacobian = Local1Jacobian(
            blocks=jnp.arange(6.0, dtype=jnp.float32).reshape(1, 1, 2, 3),
            owners=OwnerRoles(OwnerRole(0, np.array([0, 1], dtype=np.int32))),
            input_shape=(2, 1),
            input_owner_axis=0,
        )
        replacement_owners = OwnerRoles(
            OwnerRole(1, np.array([1, 0, 1], dtype=np.int32))
        )

        updated = jacobian.with_blocks(jacobian.blocks, owners=replacement_owners)

        assert updated.owners == replacement_owners

    def test_with_blocks_validates_replacement_owners(self):
        jacobian = Local1Jacobian(
            blocks=jnp.arange(6.0, dtype=jnp.float32).reshape(1, 1, 2, 3),
            owners=OwnerRoles(OwnerRole(0, np.array([0, 1], dtype=np.int32))),
            input_shape=(2, 1),
            input_owner_axis=0,
        )
        invalid_owners = OwnerRoles(OwnerRole(1, np.array([0, 1], dtype=np.int32)))

        with pytest.raises(ValueError, match="OwnerRoles shape axis 1 must be 1 or 3"):
            jacobian.with_blocks(jacobian.blocks, owners=invalid_owners)
