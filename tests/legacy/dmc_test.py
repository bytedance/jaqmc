# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc_legacy.dmc.dmc import do_run_dmc_single_walker, get_to_pad_num


def test_single_walker_across_node() -> None:
    position = np.array([1.0, 0.0, 0.0])
    walker_age = 0.0

    def dummy_wave_func(pos):
        if np.isclose(pos[0], 1.0):
            return (1, 1.0)
        return (-1, 1.0)

    def dummy_energy_func(pos):
        if np.isclose(pos[0], 1.0):
            return 0.1
        return -0.1

    def dummy_velocity(_pos):
        return jnp.ones(3) * 0.1

    (
        position_new,
        average_local_energy_new,
        walker_age_new,
        _local_energy_new,
        _weight_delta_log,
        _delta_r,
        acceptance_rate,
        debug_info,
    ) = do_run_dmc_single_walker(
        position,
        walker_age,
        0.0,
        dummy_wave_func,
        dummy_velocity,
        dummy_energy_func,
        time_step=0.01,
        key=jax.random.PRNGKey(42),
        energy_offset=0.1,
        mixed_estimator=0.1,
        nuclei=jnp.ones((1, 3)),
        charges=jnp.ones(3),
    )

    is_accepted, *_ = debug_info
    chex.assert_trees_all_close(position_new, position)
    assert float(average_local_energy_new) == pytest.approx(0.0)
    assert walker_age_new == walker_age + 1
    assert bool(is_accepted) is False
    assert float(acceptance_rate) == pytest.approx(0.0)


def test_single_walker_should_accept_diffusion() -> None:
    position = np.array([1.0, 0.0, 0.0])
    walker_age = 0.0
    dummy_velocity = jnp.ones(3) * 0.1

    def dummy_wave_func(_pos):
        return (1, 1.0)

    def dummy_energy_func(pos):
        if np.isclose(pos[0], 1.0):
            return 0.1
        return -0.1

    def dummy_velocity_func(_pos):
        return dummy_velocity

    (
        position_new,
        average_local_energy_new,
        _walker_age_new,
        _local_energy_new,
        _weight_delta_log,
        _delta_r,
        acceptance_rate,
        debug_info,
    ) = do_run_dmc_single_walker(
        position,
        walker_age,
        0.0,
        dummy_wave_func,
        dummy_velocity_func,
        dummy_energy_func,
        time_step=0.01,
        key=jax.random.PRNGKey(42),
        energy_offset=0.1,
        mixed_estimator=0.1,
        nuclei=jnp.ones((1, 3)),
        charges=jnp.ones(3),
    )

    is_accepted, *_ = debug_info
    expected_energy_new = acceptance_rate * (-0.1) + (1 - acceptance_rate) * 0.1
    delta_position_norm = jnp.linalg.norm(
        position_new - position - dummy_velocity * 0.01
    )

    assert bool(is_accepted) is True
    assert float(position_new[0]) != pytest.approx(float(position[0]))
    assert float(delta_position_norm) < float(2 * jnp.sqrt(0.01))
    assert float(average_local_energy_new) == pytest.approx(float(expected_energy_new))


def test_single_walker_should_accept_old_walker() -> None:
    position = np.array([1.0, 0.0, 0.0])
    walker_age = 0.0
    walker_age_old = 100.0
    dummy_velocity = jnp.ones(3) * 0.1

    def dummy_wave_func(pos):
        if np.isclose(pos[0], 1.0):
            return (1, 1.0)
        return (1, 0.1)

    def dummy_energy_func(_pos):
        return 0.1

    def dummy_velocity_func(_pos):
        return dummy_velocity

    *_, debug_info = do_run_dmc_single_walker(
        position,
        walker_age,
        0.0,
        dummy_wave_func,
        dummy_velocity_func,
        dummy_energy_func,
        time_step=0.01,
        key=jax.random.PRNGKey(42),
        energy_offset=0.1,
        mixed_estimator=0.1,
        nuclei=jnp.ones((1, 3)),
        charges=jnp.ones(3),
    )
    is_accepted, *_ = debug_info
    assert bool(is_accepted) is False

    (
        position_new,
        _average_local_energy_new,
        _walker_age_new,
        _local_energy_new,
        _weight_delta_log,
        _delta_r,
        acceptance_rate,
        debug_info_old,
    ) = do_run_dmc_single_walker(
        position,
        walker_age_old,
        0.0,
        dummy_wave_func,
        dummy_velocity_func,
        dummy_energy_func,
        time_step=0.01,
        key=jax.random.PRNGKey(42),
        energy_offset=0.1,
        mixed_estimator=0.1,
        nuclei=jnp.ones((1, 3)),
        charges=jnp.ones(3),
    )

    is_accepted_old, *_ = debug_info_old
    delta_position_norm = jnp.linalg.norm(
        position_new - position - dummy_velocity * 0.01
    )

    assert bool(is_accepted_old) is True
    assert float(position_new[0]) != pytest.approx(float(position[0]))
    assert float(delta_position_norm) < float(2 * jnp.sqrt(0.01))
    assert float(acceptance_rate) > 0.8


@pytest.mark.parametrize(
    ("num_walkers", "num_device", "expected_target_num"),
    [
        (4096, 8, 4160),
        (4095, 8, 4160),
        (4072, 8, 4160),
        (4071, 8, 4080),
        (40960, 8, 41600),
        (40920, 8, 41600),
        (40961, 8, 41600),
        (40800, 8, 41600),
        (40799, 8, 41600),
        (40791, 8, 40800),
    ],
)
def test_get_to_pad_num(
    num_walkers: int,
    num_device: int,
    expected_target_num: int,
) -> None:
    expected_to_pad_num = expected_target_num - num_walkers
    assert get_to_pad_num(num_walkers, num_device) == expected_to_pad_num
