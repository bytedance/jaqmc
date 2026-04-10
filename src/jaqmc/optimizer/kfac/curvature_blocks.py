# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/google-deepmind/ferminet/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""Define and register custom KFAC blocks.

As elaborated in `complex_support.py`, we should update the estimation of curvatures to
support complex numbers.
"""

from math import prod
from string import ascii_lowercase

import kfac_jax
from jax import numpy as jnp


class RepeatedDenseBlock(kfac_jax.DenseTwoKroneckerFactored):
    """Dense block that is repeatedly applied to multiple inputs (e.g. vmap).

    By default, kfac_jax will assume that the blocks only transforms the last axis,
    i.e. `ij,jk->ik`. However, in general, the dense block can be repeatedly applied
    (i.e. transforming the last axis and keeping all other axis onchanged), or we can
    transform more axis, and the output shape is not necessarily the same as the input.
    Therefore, we need to modify the ways to handle the parameters.
    """

    def __init__(self, layer_tag_eq):
        # Even though the superclass constructor will set this later, we need to do
        # it now since it's used below by `self.parameters_shapes`.
        self._layer_tag_eq = layer_tag_eq

        parameters_specs = []
        _, w_dim_out = self.weight_dimension_split

        for shape in self.parameters_shapes:
            in_str = ascii_lowercase[: len(shape)]
            # Don't use `w_dim_in` because it's not applicable to bias
            out_str = f"({in_str[:-w_dim_out]})({in_str[-w_dim_out:]})"
            parameters_specs.append(f"{in_str} -> {out_str}")

        super().__init__(layer_tag_eq, parameters_specs)

    @property
    def weight_dimension_split(self) -> tuple[int, int]:
        """Determine the input/output dimensions transformed by the weight matrix.

        The formula derives these values based on the input/output tensor dimensions
        and the weight's shape to ensure proper contraction during matrix operations.

        Returns:
            - w_dim_in: Dimensions of the input's feature space transformed by weight.
            - w_dim_out: Dimensions of the output's transformed feature space.
        """
        input_dim = len(self.inputs_shapes[0])
        output_dim = len(self.outputs_shapes[0])
        w_dim = len(self.parameters_shapes[0])
        # No further assumption. It is equired for proper contraction
        w_dim_in = (w_dim - (output_dim - input_dim)) // 2
        w_dim_out = w_dim - w_dim_in
        return w_dim_in, w_dim_out

    def fixed_scale(self) -> kfac_jax.utils.Numeric:
        """Returns a fixed scalar pre-factor of the curvature (e.g. constant)."""
        (x_shape,) = self.inputs_shapes
        return float(kfac_jax.utils.product(x_shape) // (x_shape[0] * x_shape[-1]))

    def update_curvature_matrix_estimate(
        self,
        state: kfac_jax.KroneckerFactored.State,  # type: ignore
        estimation_data: kfac_jax.LayerVjpData[kfac_jax.utils.Array],
        ema_old: kfac_jax.utils.Numeric,
        ema_new: kfac_jax.utils.Numeric,
        identity_weight: kfac_jax.utils.Numeric,
        batch_size: kfac_jax.utils.Numeric,
    ) -> kfac_jax.KroneckerFactored.State:
        """Returns the reshaped inputs and outputs.

        Also takes care of the complex conjugate.
        """
        del identity_weight
        assert 1 <= self.number_of_parameters <= 2

        # Copy this first since we mutate it later in this function.
        state = state.copy()

        [x] = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs
        assert x.shape[0] == batch_size

        w_dim_in, w_dim_out = self.weight_dimension_split
        feature_size_in = prod(x.shape[-w_dim_in:])
        feature_size_out = prod(dy.shape[-w_dim_out:])

        x = x.reshape([-1, feature_size_in])
        dy = dy.reshape([-1, feature_size_out])
        batch_size = x.size // feature_size_in
        assert all(arg.shape[0] == batch_size for arg in (x, dy))

        if self.number_of_parameters == 2:
            x_one = jnp.ones_like(x[:, :1])
            x = jnp.concatenate([x, x_one], axis=1)

        input_stats = jnp.einsum("ay,az->yz", x, x) / batch_size
        output_stats = jnp.einsum("ay,az->yz", dy.conj(), dy).real / batch_size

        state.factors[0].update(input_stats, ema_old, ema_new)
        state.factors[1].update(output_stats, ema_old, ema_new)

        return state


class DenseTwoKroneckerFactored(kfac_jax.DenseTwoKroneckerFactored):
    @kfac_jax.utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self,
        state: kfac_jax.KroneckerFactored.State,
        estimation_data: kfac_jax.tracer.LayerVjpData[kfac_jax.utils.Array],
        ema_old: kfac_jax.utils.Numeric,
        ema_new: kfac_jax.utils.Numeric,
        identity_weight: kfac_jax.utils.Numeric,
        batch_size: kfac_jax.utils.Numeric,
    ) -> kfac_jax.KroneckerFactored.State:
        del identity_weight
        assert 1 <= self.number_of_parameters <= 2

        # Copy this first since we mutate it later in this function.
        state = state.copy()

        [x] = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs

        assert kfac_jax.utils.first_dim_is_size(batch_size, x, dy)  # type: ignore

        if self.number_of_parameters == 2:
            x_one = jnp.ones_like(x[:, :1])
            x = jnp.concatenate([x, x_one], axis=1)

        input_stats = jnp.einsum("ay,az->yz", x, x) / batch_size
        # Take care of complex numbers
        output_stats = jnp.einsum("ay,az->yz", dy.conj(), dy).real / batch_size

        state.factors[0].update(input_stats, ema_old, ema_new)
        state.factors[1].update(output_stats, ema_old, ema_new)

        return state


class NaiveDiagonal(kfac_jax.NaiveDiagonal):
    @kfac_jax.utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self,
        state: kfac_jax.Diagonal.State,
        estimation_data: kfac_jax.LayerVjpData[kfac_jax.utils.Array],
        ema_old: kfac_jax.utils.Numeric,
        ema_new: kfac_jax.utils.Numeric,
        identity_weight: kfac_jax.utils.Numeric,
        batch_size: kfac_jax.utils.Numeric,
    ) -> kfac_jax.Diagonal.State:
        del identity_weight
        state = state.copy()

        for factor, dw in zip(state.diagonal_factors, estimation_data.tangents.params):
            # Take care of complex numbers
            factor.update(jnp.real(dw.conj() * dw) / batch_size, ema_old, ema_new)

        return state


class ScaleAndShiftDiagonal(kfac_jax.ScaleAndShiftDiagonal):
    @kfac_jax.utils.auto_scope_method
    def update_curvature_matrix_estimate(
        self,
        state: kfac_jax.Diagonal.State,
        estimation_data: kfac_jax.tracer.LayerVjpData[kfac_jax.utils.Array],
        ema_old: kfac_jax.utils.Numeric,
        ema_new: kfac_jax.utils.Numeric,
        identity_weight: kfac_jax.utils.Numeric,
        batch_size: kfac_jax.utils.Numeric,
    ) -> kfac_jax.Diagonal.State:
        del identity_weight

        # Copy this first since we mutate it later in this function.
        state = state.copy()

        [x] = estimation_data.primals.inputs
        [dy] = estimation_data.tangents.outputs

        assert kfac_jax.utils.first_dim_is_size(batch_size, x, dy)  # type: ignore

        if self.has_scale:
            assert state.diagonal_factors[0].shape == self.parameters_shapes[0]

            scale_shape = estimation_data.primals.params[0].shape

            d_scale = kfac_jax.curvature_blocks.utils.compatible_sum(
                x * dy, scale_shape, skip_axes=[0]
            )

            scale_diag_update = (
                jnp.sum(
                    # Take care of complex numbers
                    jnp.real(d_scale.conj() * d_scale),
                    axis=0,
                    keepdims=d_scale.ndim == len(scale_shape),
                )
                / batch_size
            )

            state.diagonal_factors[0].update(scale_diag_update, ema_old, ema_new)

        if self.has_shift:
            shift_shape = estimation_data.primals.params[-1].shape
            d_shift = kfac_jax.curvature_blocks.utils.compatible_sum(
                dy, shift_shape, skip_axes=[0]
            )

            shift_diag_update = (
                jnp.sum(
                    # Take care of complex numbers
                    jnp.real(d_shift.conj() * d_shift),
                    axis=0,
                    keepdims=d_shift.ndim == len(shift_shape),
                )
                / batch_size
            )

            state.diagonal_factors[-1].update(shift_diag_update, ema_old, ema_new)

        return state


def make_tag_to_block_ctor():
    return {
        "repeated_dense": RepeatedDenseBlock,
        "dense": DenseTwoKroneckerFactored,
        "generic": NaiveDiagonal,
        "scale_and_shift": ScaleAndShiftDiagonal,
    }
