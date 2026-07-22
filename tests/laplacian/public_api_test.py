# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Public API tests for Forward Laplacian inputs, seeds, and guard helpers."""

import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqmc.laplacian import (
    LapTuple,
    Local1Jacobian,
    Local2Jacobian,
    OwnerRole,
    OwnerRoles,
    forward_laplacian,
    is_dense_laptuple,
    is_laptuple,
    is_local1_laptuple,
    is_local2_laptuple,
    is_sparse_jacobian,
    is_sparse_laptuple,
    make_laplacian_input,
)
from tests.laplacian.helpers import (
    assert_allclose,
    check_with_brute_force,
)


class TestLapTuple:
    def test_shape_delegation(self):
        x = jnp.ones((3, 4), dtype=jnp.float32)
        arr = LapTuple(
            x=x,
            jacobian=jnp.zeros((3, 4, 5), dtype=jnp.float32),
            laplacian=jnp.zeros((3, 4), dtype=jnp.float32),
        )
        assert arr.shape == (3, 4)
        assert arr.ndim == 2
        assert arr.size == 12
        assert arr.dtype == jnp.float32

    @pytest.mark.requires_x64
    def test_astype_float(self):
        arr = LapTuple(
            x=jnp.array([1.0], dtype=jnp.float32),
            jacobian=jnp.array([[2.0]], dtype=jnp.float32),
            laplacian=jnp.array([3.0], dtype=jnp.float32),
        )
        result = arr.astype(jnp.float64)
        expected_dtype = jax.dtypes.canonicalize_dtype(jnp.float64)
        assert isinstance(result, LapTuple)
        assert result.x.dtype == expected_dtype
        assert result.jacobian.dtype == expected_dtype
        assert result.laplacian.dtype == expected_dtype

    def test_astype_int_drops_derivatives(self):
        arr = LapTuple(
            x=jnp.array([1.5]),
            jacobian=jnp.array([[2.0]]),
            laplacian=jnp.array([3.0]),
        )
        result = arr.astype(jnp.int32)
        assert not isinstance(result, LapTuple)
        assert result.dtype == jnp.int32
        assert jnp.array_equal(result, jnp.array([1]))

    def test_astype_bool_drops_derivatives(self):
        arr = LapTuple(
            x=jnp.array([0.0, 1.5]),
            jacobian=jnp.ones((3, 2)),
            laplacian=jnp.zeros(2),
        )
        result = arr.astype(jnp.bool_)
        assert not isinstance(result, LapTuple)
        assert result.dtype == jnp.bool_
        assert jnp.array_equal(result, jnp.array([False, True]))

    def test_pytree_spec_preserves_laptuple_node(self):
        spec = LapTuple.pytree_spec(0, 1, 0)
        assert isinstance(spec, LapTuple)
        assert spec.x == 0
        assert spec.jacobian == 1
        assert spec.laplacian == 0

        value = LapTuple(
            jnp.array(1.0),
            jnp.array([[2.0]]),
            jnp.array(3.0),
        )
        value_treedef = jax.tree_util.tree_structure(value)
        reconstructed = jax.tree_util.tree_unflatten(value_treedef, [0, 1, 0])
        assert isinstance(reconstructed, LapTuple)
        assert reconstructed.x == 0
        assert reconstructed.jacobian == 1
        assert reconstructed.laplacian == 0

    def test_dense_jacobian_materializes_compact_scalar_output(self):
        """Compact scalar-output derivatives broadcast over the primal shape."""
        x = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)
        arr = LapTuple(
            x=x,
            jacobian=jnp.arange(x.size, dtype=x.dtype),
            laplacian=jnp.zeros_like(x),
        )

        expected = jnp.broadcast_to(
            jnp.arange(x.size, dtype=x.dtype).reshape(x.size, 1, 1),
            (x.size, *x.shape),
        )

        assert_allclose(arr.dense_jacobian, expected)


class TestMakeLaplacianInputDense:
    def test_identity(self):
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        seed = make_laplacian_input(x)
        expected = jnp.eye(x.size, dtype=x.dtype).reshape(x.size, *x.shape)
        assert_allclose(seed.jacobian, expected)
        assert_allclose(seed.laplacian, jnp.zeros_like(x))

    def test_weights(self):
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        weights = jnp.array([[1.0, 2.0, 3.0], [0.5, 0.25, 0.125]], dtype=x.dtype)
        seed = make_laplacian_input(x, weights=weights)
        expected = jnp.eye(x.size, dtype=x.dtype).reshape(x.size, *x.shape)
        expected = expected * weights
        assert_allclose(seed.jacobian, expected)

    def test_pytree_weights(self):
        x = {
            "a": jnp.array([1.0, 2.0], dtype=jnp.float32),
            "b": (jnp.array([[3.0]], dtype=jnp.float32),),
        }
        weights = {
            "a": jnp.array([2.0, 0.5], dtype=jnp.float32),
            "b": (jnp.array([[1.5]], dtype=jnp.float32),),
        }

        seed = make_laplacian_input(x, weights=weights)

        assert isinstance(seed["a"], LapTuple)
        assert isinstance(seed["b"][0], LapTuple)
        expected_a = jnp.array([[2.0, 0.0], [0.0, 0.5], [0.0, 0.0]], dtype=jnp.float32)
        expected_b = jnp.array([[[0.0]], [[0.0]], [[1.5]]], dtype=jnp.float32)
        assert_allclose(seed["a"].jacobian, expected_a)
        assert_allclose(seed["b"][0].jacobian, expected_b)


class TestMakeLaplacianInputSparse:
    def test_axis_rejects_out_of_bounds(self):
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

        with pytest.raises(ValueError, match="axis 2 is out of bounds for ndim 2"):
            make_laplacian_input(x, sparse_axis=2)

    def test_weights_match_dense_identity(self):
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        weights = jnp.array([[1.0, 2.0, 3.0], [0.5, 0.25, 0.125]], dtype=x.dtype)

        dense = make_laplacian_input(x, weights=weights)
        sparse = make_laplacian_input(x, weights=weights, sparse_axis=0)

        assert_allclose(sparse.dense_jacobian, dense.dense_jacobian)
        assert_allclose(sparse.laplacian, dense.laplacian)

    def test_rejects_pytree(self):
        with pytest.raises(TypeError, match="requires a single array input"):
            make_laplacian_input(
                (jnp.array([1.0], dtype=jnp.float32),),  # type: ignore
                sparse_axis=0,
            )

    @pytest.mark.parametrize(
        "x",
        (
            pytest.param(jnp.arange(4.0, dtype=jnp.float32), id="vector"),
            pytest.param(
                jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
                id="matrix",
            ),
            pytest.param(
                jnp.arange(8.0, dtype=jnp.float32).reshape(2, 2, 2),
                id="packed_coords",
            ),
        ),
    )
    def test_identity_materialization_matches_dense(self, x):
        """Sparse Local1 seeding materializes the dense identity basis."""
        dense = make_laplacian_input(x)
        sparse = make_laplacian_input(x, sparse_axis=0)

        assert isinstance(sparse.jacobian, Local1Jacobian)
        assert_allclose(sparse.dense_jacobian, dense.dense_jacobian)
        assert_allclose(sparse.laplacian, jnp.zeros_like(x))

    def test_identity_materialization_for_nonzero_owner_axis(self):
        x = jnp.arange(24.0, dtype=jnp.float32).reshape(3, 2, 4)

        dense = make_laplacian_input(x)
        sparse = make_laplacian_input(x, sparse_axis=1)

        assert sparse.jacobian.input_owner_axis == 1
        assert_allclose(sparse.dense_jacobian, dense.dense_jacobian)

    def test_negative_owner_axis_matches_positive_axis(self):
        x = jnp.arange(24.0, dtype=jnp.float32).reshape(3, 2, 4)

        negative_axis = make_laplacian_input(x, sparse_axis=-2)
        positive_axis = make_laplacian_input(x, sparse_axis=1)

        assert negative_axis.jacobian.input_owner_axis == 1
        assert_allclose(negative_axis.dense_jacobian, positive_axis.dense_jacobian)


class TestInputSeeding:
    @staticmethod
    def _auto_seed(x: jnp.ndarray) -> LapTuple:
        return make_laplacian_input(x)

    @staticmethod
    def _manual_identity_seed(x: jnp.ndarray) -> LapTuple:
        return LapTuple(
            x,
            jnp.eye(x.size, dtype=x.dtype).reshape(x.size, *x.shape),
            jnp.zeros_like(x),
        )

    @pytest.mark.parametrize(
        "build_seed",
        (
            pytest.param(_auto_seed, id="make_laplacian_input"),
            pytest.param(_manual_identity_seed, id="manual_identity_laptuple"),
        ),
    )
    def test_explicit_dense_seed_matches_auto_seed(self, build_seed):
        """Raw-array auto-seeding matches explicit dense identity seeding."""
        x = jnp.array([[0.5, -0.3], [0.8, 0.4]], dtype=jnp.float32)

        def fn(value):
            return jnp.sum(jnp.sin(value) + value**3)

        fl = forward_laplacian(fn)
        auto = fl(x)
        seeded = fl(build_seed(x))

        assert_allclose(auto.x, seeded.x)
        assert_allclose(auto.dense_jacobian, seeded.dense_jacobian)
        assert_allclose(auto.laplacian, seeded.laplacian)


class TestGuards:
    def test_top_level_guard_helpers(self):
        dense = LapTuple(
            jnp.array([1.0, 2.0]),
            jnp.eye(2, dtype=jnp.float32),
            jnp.zeros(2, dtype=jnp.float32),
        )
        local1 = make_laplacian_input(
            jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3),
            sparse_axis=0,
        )
        owners = OwnerRoles(
            OwnerRole(0, np.array([0, 1], dtype=np.int32)),
            OwnerRole(1, np.array([0, 1], dtype=np.int32)),
        )
        local2 = LapTuple(
            jnp.zeros((2, 2, 3), dtype=jnp.float32),
            Local2Jacobian(
                blocks=jnp.zeros((2, 3, 2, 2, 3), dtype=jnp.float32),
                owners=owners,
                input_shape=(2, 3),
                input_owner_axis=0,
            ),
            jnp.zeros((2, 2, 3), dtype=jnp.float32),
        )

        assert is_laptuple(dense)
        assert is_dense_laptuple(dense)
        assert not is_sparse_laptuple(dense)
        assert is_local1_laptuple(local1)
        assert is_sparse_laptuple(local1)
        assert is_sparse_jacobian(local1.jacobian)
        assert isinstance(local1.jacobian, Local1Jacobian)
        assert is_local2_laptuple(local2)
        assert is_sparse_laptuple(local2)
        assert is_sparse_jacobian(local2.jacobian)
        assert not is_laptuple(jnp.array([1.0]))


class TestDenseLapTupleContract:
    def test_forward_laplacian_rejects_mismatched_dense_bases(self):
        fl = forward_laplacian(operator.add)
        lhs = LapTuple(jnp.array([1.0]), jnp.ones((2, 1)), jnp.zeros(1))
        rhs = LapTuple(jnp.array([2.0]), jnp.ones((3, 1)), jnp.zeros(1))

        with pytest.raises(
            ValueError,
            match="Dense LapTuple Jacobians must share the same tracked-input basis",
        ):
            fl(lhs, rhs)


class TestLapTuplePassIn:
    """Passing pre-built derivative state through the public transform."""

    def test_weighted_jacobian(self):
        """Weighted initial Jacobian computes tr(W H W^T)."""
        x = jnp.array([1.0, 2.0, 3.0])
        w = jnp.array([2.0, 0.5, 1.0])
        fn = lambda value: jnp.sum(value**3)
        check_with_brute_force(fn, LapTuple(x, jnp.diag(w), jnp.zeros(3)))

    def test_composition(self):
        """Manual LapTuple outputs can be passed into another public transform."""
        x = jnp.array([1.0, 2.0, 3.0])
        fl_f = forward_laplacian(jnp.sin)
        fl_g = forward_laplacian(lambda value: jnp.sum(value**2))
        intermediate = fl_f(x)
        composed = fl_g(intermediate)
        check_with_brute_force(
            lambda value: jnp.sum(jnp.sin(value) ** 2),
            x,
            actual_result=composed,
        )

    def test_mixed_laptuple_and_plain(self):
        """A manual LapTuple makes sibling plain arrays constants."""
        x = jnp.array([1.0, 2.0])
        c = jnp.array([3.0, 4.0])
        fn = lambda value, scale: jnp.sum(value**2 * scale)
        x_lapl = LapTuple(x, jnp.eye(2), jnp.zeros(2))
        check_with_brute_force(fn, x_lapl, c)

    def test_nonzero_initial_laplacian(self):
        """A manual LapTuple can provide a nonzero initial Laplacian."""
        x = jnp.array([1.0, 2.0, 3.0])
        init_lapl = jnp.array([0.5, -0.5, 1.0])
        check_with_brute_force(
            lambda value: 2 * value,
            LapTuple(x, jnp.eye(3), init_lapl),
        )


class TestSparseSeededTransform:
    def test_local1_weighted_seed_matches_brute_force(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        weights = jnp.array([[2.0, 0.5], [1.5, 0.25]])
        fn = lambda value: jnp.sum(value**3)
        sparse_input = make_laplacian_input(x, weights=weights, sparse_axis=0)
        check_with_brute_force(fn, sparse_input)
