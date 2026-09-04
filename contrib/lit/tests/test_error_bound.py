# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="operator"

"""Numerical stability tests for the source-sampled LIT error factor."""

from functools import reduce
from operator import itemgetter

import jax
import numpy as np
from jaqmc_contrib_lit.response import (
    LITSourceSums,
    _source_sampled_error_d_sums,
    merge_source_sums,
    merge_source_sums_across_devices,
    merge_weighted_complex_moments,
    stats_from_source_sums,
    weighted_complex_moments,
)
from jax import numpy as jnp

from jaqmc.utils import parallel_jax


def _source_sums(response_values, hbar_values, weights, normalization):
    response_values = jnp.asarray(response_values, dtype=jnp.complex64)
    hbar_values = jnp.asarray(hbar_values, dtype=jnp.complex64)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    normalization = jnp.asarray(normalization, dtype=jnp.complex64)
    weight_sum = jnp.sum(weights)
    ratio_scale = jnp.where(jnp.abs(normalization) > 0.0, jnp.abs(normalization), 1.0)
    scaled_normalization = normalization / ratio_scale
    ratio_abs2_sum = weight_sum * jnp.abs(scaled_normalization) ** 2
    return LITSourceSums(
        sample_count=jnp.asarray(weights.size, dtype=jnp.float32),
        weight_sum=weight_sum,
        valid_sample_count=jnp.sum(weights > 0.0),
        ratio_scale=ratio_scale,
        ratio_sum=weight_sum * scaled_normalization,
        ratio_abs2_sum=ratio_abs2_sum,
        psi_weight_sum=ratio_abs2_sum,
        psi_weight_sq_sum=jnp.sum((weights * jnp.abs(scaled_normalization) ** 2) ** 2),
        psi_log_ratio_abs2_sum=jnp.asarray(0.0, dtype=jnp.float32),
        response_conj_over_source_sum=jnp.sum(weights * jnp.conj(response_values)),
        ground_energy_sum=jnp.asarray(0.0, dtype=jnp.float32),
        response_over_source_moments=weighted_complex_moments(
            response_values,
            weights,
        ),
        hbar_over_source_moments=weighted_complex_moments(hbar_values, weights),
        psi_weight_max=jnp.max(weights * jnp.abs(scaled_normalization) ** 2),
    )


def _error_result(sums, *, phi_norm=1.0, omega=0.4, eta=0.3):
    return _source_sampled_error_d_sums(
        sums,
        jnp.asarray(phi_norm, dtype=jnp.float32),
        omega=omega,
        eta=eta,
    )


def test_tiny_normalization_regression_does_not_collapse_to_zero():
    normalization = np.complex64(2.85e-7)
    shift_norm = 0.5
    response = normalization * np.asarray([0.8, 1.2], dtype=np.complex64)
    hbar = normalization * shift_norm * np.asarray([0.7, 1.3], dtype=np.complex64)
    sums = _source_sums(response, hbar, [1.0, 1.0], normalization)

    stats = stats_from_source_sums(
        sums,
        source_norm=1.0,
        omega=0.4,
        eta=0.3,
    )

    assert bool(stats.error_d_valid)
    normalization_abs = abs(normalization)
    np.testing.assert_allclose(
        float(stats.error_d_correction),
        0.2 * normalization_abs,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        float(stats.error_d_shifted),
        0.3 * normalization_abs,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        float(stats.error_d),
        0.2 * normalization_abs,
        rtol=2e-6,
    )
    assert float(stats.error_d) > 0.0
    np.testing.assert_allclose(float(stats.reweight_max_fraction), 0.5, rtol=2e-6)
    np.testing.assert_allclose(
        float(stats.correction_norm),
        1.04 * normalization_abs**2,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        float(stats.shifted_hamiltonian_norm),
        1.09 * normalization_abs**2,
        rtol=2e-6,
    )


def test_raw_error_d_matches_supplement_eq19_by_hand():
    weights = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    response = np.asarray([0.3 + 0.2j, 1.1 - 0.4j, -0.2 + 0.7j], dtype=np.complex64)
    hbar = np.asarray([-0.5 + 0.1j, 0.6 + 0.8j, 1.2 - 0.3j], dtype=np.complex64)
    phi_norm = 2.25
    omega = 0.3
    eta = 0.4
    shift_abs = np.hypot(omega, eta)
    sums = _source_sums(response, hbar, weights, normalization=0.9 - 0.1j)

    result = _error_result(
        sums,
        phi_norm=phi_norm,
        omega=omega,
        eta=eta,
    )
    weight_sum = np.sum(weights, dtype=np.float64)
    response_mean = np.sum(weights * response, dtype=np.complex128) / weight_sum
    hbar_mean = np.sum(weights * hbar, dtype=np.complex128) / weight_sum
    response_m2 = np.sum(weights * np.abs(response - response_mean) ** 2)
    hbar_m2 = np.sum(weights * np.abs(hbar - hbar_mean) ** 2)
    expected_correction_norm = phi_norm * (
        response_m2 / weight_sum + abs(response_mean) ** 2
    )
    expected_shifted_norm = (
        phi_norm * (hbar_m2 / weight_sum + abs(hbar_mean) ** 2) / shift_abs**2
    )
    expected_d1 = np.sqrt(phi_norm * response_m2 / weight_sum)
    expected_d2 = np.sqrt(phi_norm * hbar_m2 / weight_sum) / shift_abs

    assert bool(result[-1])
    np.testing.assert_allclose(float(result[0]), expected_correction_norm, rtol=2e-6)
    np.testing.assert_allclose(float(result[1]), expected_shifted_norm, rtol=2e-6)
    np.testing.assert_allclose(float(result[2]), expected_d1, rtol=2e-6)
    np.testing.assert_allclose(float(result[3]), expected_d2, rtol=2e-6)
    np.testing.assert_allclose(
        float(result[4]), min(expected_d1, expected_d2), rtol=2e-6
    )


def test_raw_error_d_scales_under_global_complex_rescaling():
    base_normalization = np.complex64(0.7 - 0.2j)
    response_factors = np.asarray([0.4 + 0.1j, 1.1 - 0.2j, 1.8 + 0.3j])
    hbar_factors = np.asarray([0.2 - 0.4j, 0.9 + 0.1j, 1.3 + 0.5j])
    weights = np.asarray([1.0, 3.0, 2.0], dtype=np.float32)
    reference_norms = None
    reference_d = None

    for magnitude in (1.0, 1e-12, 1e-6, 1e6, 1e12):
        scale = np.complex64(magnitude * np.exp(0.37j))
        normalization = scale * base_normalization
        sums = _source_sums(
            scale * base_normalization * response_factors,
            scale * base_normalization * 0.5 * hbar_factors,
            weights,
            normalization,
        )
        result = _error_result(sums)
        assert bool(result[-1])
        current_norms = np.asarray(result[:2], dtype=np.float64)
        current_d = np.asarray(result[2:5], dtype=np.float64)
        assert np.all(np.isfinite(current_norms))
        assert np.all(np.isfinite(current_d))
        if reference_d is None:
            reference_norms = current_norms
            reference_d = current_d
        else:
            np.testing.assert_allclose(
                current_norms,
                reference_norms * magnitude**2,
                rtol=4e-6,
            )
            np.testing.assert_allclose(
                current_d,
                reference_d * magnitude,
                rtol=4e-6,
            )


def test_signed_lit_has_no_absolute_normalization_floor():
    base_normalization = np.complex64(0.7 - 0.2j)
    response = np.asarray(
        [0.3 + 0.2j, 1.1 - 0.4j, -0.2 + 0.7j],
        dtype=np.complex64,
    )
    weights = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    reference = None

    for magnitude in (1.0, 1e-20, 1e-8, 1e8, 1e20):
        scale = np.complex64(magnitude * np.exp(0.37j))
        sums = _source_sums(
            scale * response,
            0.5 * scale * response,
            weights,
            scale * base_normalization,
        )
        stats = stats_from_source_sums(
            sums,
            source_norm=1.7,
            omega=0.4,
            eta=0.3,
        )
        current = np.asarray(
            [stats.signed_lit, stats.broadened, stats.equation_relative_residual],
            dtype=np.float64,
        )
        assert np.all(np.isfinite(current))
        if reference is None:
            reference = current
        else:
            np.testing.assert_allclose(current, reference, rtol=8e-6, atol=2e-6)


def test_zero_normalization_marks_normalized_observables_invalid():
    sums = _source_sums(
        [0.3 + 0.2j, 1.1 - 0.4j],
        [0.1 + 0.1j, 0.5 - 0.2j],
        [1.0, 2.0],
        normalization=0.0 + 0.0j,
    )
    stats = stats_from_source_sums(
        sums,
        source_norm=1.0,
        omega=0.4,
        eta=0.3,
    )

    assert np.isnan(float(stats.signed_lit))
    assert np.isnan(float(stats.broadened))
    assert np.isnan(float(stats.equation_relative_residual))


def test_near_collinear_float32_moment_retains_small_fluctuations():
    values = np.asarray(
        [
            1_000_000.0 + 200_000.0j,
            1_000_000.125 + 199_999.96875j,
            999_999.8125 + 200_000.0625j,
            1_000_000.25 + 199_999.9375j,
        ],
        dtype=np.complex64,
    )
    weights = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    moments = weighted_complex_moments(jnp.asarray(values), jnp.asarray(weights))
    values_128 = values.astype(np.complex128)
    weights_64 = weights.astype(np.float64)
    origin = values_128[0]
    expected_mean = origin + np.sum(weights_64 * (values_128 - origin)) / np.sum(
        weights_64
    )
    expected_m2 = np.sum(weights_64 * np.abs(values_128 - expected_mean) ** 2)

    assert float(moments.centered_abs2_sum) > 0.0
    np.testing.assert_allclose(
        complex(moments.mean),
        expected_mean,
        rtol=0.0,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        float(moments.centered_abs2_sum),
        expected_m2,
        rtol=1e-5,
    )


def test_weighted_moments_handle_unequal_and_zero_weights():
    values = np.asarray([10.0 + 2.0j, 1.0 - 3.0j, 4.0 + 5.0j, -8.0j])
    weights = np.asarray([0.0, 1.0, 9.0, 0.0], dtype=np.float32)
    moments = weighted_complex_moments(jnp.asarray(values), jnp.asarray(weights))
    expected_mean = np.sum(weights * values) / np.sum(weights)
    expected_m2 = np.sum(weights * np.abs(values - expected_mean) ** 2)

    np.testing.assert_allclose(float(moments.weight_sum), 10.0)
    np.testing.assert_allclose(complex(moments.mean), expected_mean, rtol=2e-7)
    np.testing.assert_allclose(
        float(moments.centered_abs2_sum),
        expected_m2,
        rtol=2e-7,
    )

    empty = weighted_complex_moments(jnp.asarray(values), jnp.zeros(4))
    np.testing.assert_allclose(float(empty.weight_sum), 0.0)
    np.testing.assert_allclose(complex(empty.mean), 0.0)
    np.testing.assert_allclose(float(empty.centered_abs2_sum), 0.0)


def test_chan_chunk_merge_matches_direct_centered_moment_and_error_d():
    normalization = np.complex64(0.8 + 0.1j)
    response = normalization * np.asarray(
        [0.7 + 0.2j, 1.0 - 0.1j, 1.4 + 0.3j, 0.9 - 0.4j, 1.2 + 0.1j]
    )
    hbar = (
        normalization
        * 0.5
        * np.asarray([0.2 + 0.1j, 1.1 - 0.3j, 0.8 + 0.2j, 1.5 + 0.4j, 0.6 - 0.2j])
    )
    weights = np.asarray([0.0, 1.0, 7.0, 2.0, 0.5], dtype=np.float32)
    full = _source_sums(response, hbar, weights, normalization)
    chunks = [
        _source_sums(response[:1], hbar[:1], weights[:1], normalization),
        _source_sums(response[1:3], hbar[1:3], weights[1:3], normalization),
        _source_sums(response[3:], hbar[3:], weights[3:], normalization),
    ]
    merged = reduce(merge_source_sums, chunks)

    for merged_moments, full_moments in (
        (merged.response_over_source_moments, full.response_over_source_moments),
        (merged.hbar_over_source_moments, full.hbar_over_source_moments),
    ):
        np.testing.assert_allclose(
            complex(merged_moments.mean),
            complex(full_moments.mean),
            rtol=2e-7,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            float(merged_moments.centered_abs2_sum),
            float(full_moments.centered_abs2_sum),
            rtol=2e-6,
        )

    merged_result = _error_result(merged)
    full_result = _error_result(full)
    np.testing.assert_allclose(
        np.asarray(merged_result[:5]),
        np.asarray(full_result[:5]),
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        float(merged.psi_weight_max),
        float(full.psi_weight_max),
        rtol=2e-6,
    )


def test_data_parallel_chan_merge_matches_high_precision_shard_reference():
    device_count = jax.local_device_count()
    normalization = np.complex64(0.6 - 0.15j)
    shards = []
    for device in range(device_count):
        response = np.asarray(
            [
                1_000_000.0 + 200_000.0j + 0.125 * device,
                1_000_000.125 + 199_999.96875j - 0.0625 * device,
            ],
            dtype=np.complex64,
        )
        hbar = np.asarray(
            [
                -700_000.0 + 300_000.0j + 0.0625j * device,
                -699_999.875 + 299_999.9375j - 0.125j * device,
            ],
            dtype=np.complex64,
        )
        shards.append(
            _source_sums(
                response,
                hbar,
                np.asarray([device + 1.0, 2.0], dtype=np.float32),
                normalization,
            )
        )
    sharded = jax.tree.map(lambda *leaves: jnp.stack(leaves), *shards)
    serial = reduce(merge_source_sums, shards)

    def reference_moments(field):
        moments = [getattr(shard, field) for shard in shards]
        weights = np.asarray(
            [float(moment.weight_sum) for moment in moments],
            dtype=np.float64,
        )
        means = np.asarray(
            [
                complex(moment.origin) + complex(moment.mean_offset)
                for moment in moments
            ],
            dtype=np.complex128,
        )
        centered = np.asarray(
            [float(moment.centered_abs2_sum) for moment in moments],
            dtype=np.float64,
        )
        nonempty = weights > 0.0
        origin = means[np.flatnonzero(nonempty)[0]]
        total_weight = np.sum(weights)
        mean = origin + np.sum(weights * (means - origin)) / total_weight
        centered_abs2_sum = np.sum(centered + weights * np.abs(means - mean) ** 2)
        return mean, centered_abs2_sum

    response_reference = reference_moments("response_over_source_moments")
    hbar_reference = reference_moments("hbar_over_source_moments")
    np.testing.assert_allclose(
        float(serial.response_over_source_moments.centered_abs2_sum),
        response_reference[1],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        float(serial.hbar_over_source_moments.centered_abs2_sum),
        hbar_reference[1],
        rtol=2e-6,
    )

    def merge(local_arrays):
        local_sums = jax.tree.map(itemgetter(0), local_arrays)
        return merge_source_sums_across_devices(
            local_sums,
            axis_name=parallel_jax.BATCH_AXIS_NAME,
        )

    input_specs = jax.tree.map(lambda _: parallel_jax.DATA_PARTITION, sharded)
    output_specs = jax.tree.map(lambda _: parallel_jax.SHARE_PARTITION, serial)
    merge_sharded = parallel_jax.jit_sharded(
        merge,
        in_specs=(input_specs,),
        out_specs=output_specs,
    )
    (sharded_input,) = jax.device_put(
        (sharded,),
        parallel_jax.make_sharding((input_specs,)),
    )
    parallel_result = merge_sharded(sharded_input)

    np.testing.assert_allclose(
        complex(parallel_result.response_over_source_moments.mean),
        response_reference[0],
        rtol=0.0,
        atol=0.1,
    )
    assert float(parallel_result.response_over_source_moments.centered_abs2_sum) > 0.0
    np.testing.assert_allclose(
        float(parallel_result.response_over_source_moments.centered_abs2_sum),
        response_reference[1],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        float(parallel_result.hbar_over_source_moments.centered_abs2_sum),
        hbar_reference[1],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        float(parallel_result.psi_weight_max),
        float(serial.psi_weight_max),
        rtol=2e-6,
    )


def test_error_d_uses_smaller_branch():
    normalization = np.complex64(0.75)
    response = normalization * np.asarray([0.6, 1.4], dtype=np.complex64)
    hbar = normalization * 0.5 * np.asarray([0.9, 1.1], dtype=np.complex64)
    result = _error_result(
        _source_sums(response, hbar, [1.0, 1.0], normalization),
    )

    np.testing.assert_allclose(float(result[2]), 0.3, rtol=2e-6)
    np.testing.assert_allclose(float(result[3]), 0.075, rtol=2e-6)
    np.testing.assert_allclose(float(result[4]), 0.075, rtol=2e-6)


def test_invalid_moments_never_report_zero_error():
    normalization = np.complex64(0.8)
    sums = _source_sums(
        normalization * np.asarray([0.8, 1.2]),
        normalization * 0.5 * np.asarray([0.7, 1.3]),
        [1.0, 1.0],
        normalization,
    )

    invalid_moments = sums.response_over_source_moments._replace(
        origin=jnp.asarray(jnp.nan + 0.0j, dtype=jnp.complex64)
    )
    invalid_sums = sums._replace(response_over_source_moments=invalid_moments)
    result = _error_result(invalid_sums)
    assert not bool(result[-1])
    assert np.isnan(float(result[4]))


def test_chan_merge_with_empty_side_preserves_nonempty_moments():
    values = jnp.asarray([1.0 + 2.0j, 3.0 - 1.0j], dtype=jnp.complex64)
    nonempty = weighted_complex_moments(values, jnp.asarray([1.0, 2.0]))
    empty = weighted_complex_moments(values, jnp.zeros(2))

    for merged in (
        merge_weighted_complex_moments(empty, nonempty),
        merge_weighted_complex_moments(nonempty, empty),
    ):
        np.testing.assert_allclose(float(merged.weight_sum), float(nonempty.weight_sum))
        np.testing.assert_allclose(complex(merged.mean), complex(nonempty.mean))
        np.testing.assert_allclose(
            float(merged.centered_abs2_sum),
            float(nonempty.centered_abs2_sum),
        )
