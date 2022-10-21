# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from absl.testing import absltest
import jax.test_util as jtu
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

from jaqmc.dmc.position_update import *

class PositionUpdateTest(jtu.JaxTestCase):
    def test_velocity_clipping_mild_v(self):
        velocity = jnp.array([0.3, 0.5, 0.2])
        position = jnp.array([1.0, 0, 0])
        nearest_nucleus = jnp.array([2.0, 0, 0])
        nearest_charge = 1.0
        time_step = 1e-2

        clipped_velocity = clip_velocity_helper(
            velocity, position, nearest_nucleus, nearest_charge, time_step)
        self.assertArraysAllClose(velocity, clipped_velocity, rtol=1e-3)

    def test_velocity_clipping_outrageous_v(self):
        velocity = jnp.array([10, 0, 0])
        position = jnp.array([1.0, 0, 0])
        nearest_nucleus = jnp.array([0.0, 0, 0])
        nearest_charge = 10.0
        time_step = 1e-1

        a = 1 + 100 / 1040
        v_norm = np.linalg.norm(velocity)
        expected_clip_factor = (-1 + np.sqrt(1 + 2 * a * v_norm**2 * time_step)) / (a * v_norm ** 2 * time_step)

        clipped_velocity = clip_velocity_helper(
            velocity, position, nearest_nucleus, nearest_charge, time_step)
        self.assertArraysAllClose(expected_clip_factor * velocity, clipped_velocity)


    def test_do_drift_mild(self):
        position = jnp.array([1.0, 0, 0])
        nearest_nucleus = jnp.array([2.0, 0, 0])
        clipped_velocity = jnp.array([0.1, 0.2, 0.3])
        time_step = 1e-2
        drifted_position = do_drift(position, nearest_nucleus, clipped_velocity, time_step)
        naive_drift_position = position + clipped_velocity * time_step
        self.assertArraysAllClose(drifted_position, naive_drift_position, rtol=1e-3)

    def test_do_drift_overshoot(self):
        position = jnp.array([1.9, 0, 0])
        nearest_nucleus = jnp.array([2.0, 0, 0])
        clipped_velocity = jnp.array([2.0, 2, 3])
        time_step = 1e-1
        drifted_position = do_drift(position, nearest_nucleus, clipped_velocity, time_step)
        expected_position = nearest_nucleus
        self.assertArraysAllClose(drifted_position, nearest_nucleus)

    def test_do_drift_close_to_overshoot(self):
        position = jnp.array([1, 0, 0])
        nearest_nucleus = jnp.array([2.0, 0, 0])
        clipped_velocity = jnp.array([9, 2, 3])
        time_step = 1e-1
        drifted_position = do_drift(position, nearest_nucleus, clipped_velocity, time_step)
        rho_factor = (2 * 0.1 / 1.1)
        expected_drift_position = position + jnp.array((clipped_velocity[0], ) + tuple(clipped_velocity[1:] * rho_factor)) * time_step
        naive_drift_position = position + clipped_velocity * time_step
        self.assertArraysAllClose(drifted_position, expected_drift_position)
        self.assertNotEqual(
            jnp.linalg.norm(naive_drift_position[1:]),
            jnp.linalg.norm(drifted_position[1:]))


    def test_overshoot_prob_small(self):
        position = jnp.array([1, 0, 0])
        nearest_nucleus = jnp.array([2.0, 0, 0])
        clipped_velocity = jnp.array([1., 2, 3])
        time_step = 1e-2
        overshoot_prob = calc_nucleus_overshoot_prob_helper(
            position,
            nearest_nucleus,
            clipped_velocity,
            time_step)
        self.assertLess(overshoot_prob, 0.1)


    def test_overshoot_prob_large(self):
        position = jnp.array([1.9, 0, 0])
        nearest_nucleus = jnp.array([2.0, 0, 0])
        clipped_velocity = jnp.array([100., 2, 3])
        time_step = 1e-2
        overshoot_prob = calc_nucleus_overshoot_prob_helper(
            position,
            nearest_nucleus,
            clipped_velocity,
            time_step)
        self.assertGreater(overshoot_prob, 0.9)

    def test_overshoot_prob_exact(self):
        position = jnp.array([0.9, 0, 0])
        nearest_nucleus = jnp.array([0.0, 0, 0])
        clipped_velocity = jnp.array([1., 2, 3])
        time_step = 0.1
        overshoot_prob = calc_nucleus_overshoot_prob_helper(
            position,
            nearest_nucleus,
            clipped_velocity,
            time_step)
        expected_prob = 0.5 * jax.lax.erfc(1 / jnp.sqrt(0.2))
        self.assertEqual(overshoot_prob, expected_prob)

    def test_do_diffusion_gaussian(self):
        drifted_position = jnp.array([10.0, 0, 0])
        nearest_nucleus = jnp.array([-10.0, 0, 0])
        overshoot_prob = 0

        seed = 42
        key = jax.random.PRNGKey(seed)

        num_data = int(1e4)
        all_keys = jax.random.split(key, num_data)
        time_step = 1e-2
        laplace_zeta = 1 / jnp.sqrt(time_step)
        vmapped_do_diffusion = jax.vmap(
            lambda key: do_diffusion(drifted_position, nearest_nucleus,
                                     overshoot_prob,
                                     key,
                                     time_step,
                                     laplace_zeta)[0])
        diffused_position = vmapped_do_diffusion(all_keys)
        gaussian_vars = (
            (diffused_position - drifted_position) / jnp.sqrt(time_step)
        ).reshape((-1,))

        def expected_sample_gaussian(size):
            return np.random.normal(size=size)
        test_result = sp.stats.kstest(expected_sample_gaussian, gaussian_vars)
        self.assertGreater(test_result.pvalue, 0.1)

    def test_do_diffusion_gamma(self):
        drifted_position = jnp.array([10.0, 0, 0])
        nearest_nucleus = jnp.array([-10.0, 0, 0])
        overshoot_prob = 1

        seed = 42
        key = jax.random.PRNGKey(seed)

        num_data = int(1e6)
        all_keys = jax.random.split(key, num_data)
        time_step = 1e-2
        laplace_zeta = 1 / jnp.sqrt(time_step)
        vmapped_do_diffusion = jax.vmap(
            lambda key: do_diffusion(drifted_position, nearest_nucleus,
                                     overshoot_prob,
                                     key,
                                     time_step,
                                     laplace_zeta)[0])
        diffused_position = vmapped_do_diffusion(all_keys)
        gamma_vars = (diffused_position - nearest_nucleus)

        self.do_test_gamma(gamma_vars, laplace_zeta)

    def do_test_gamma(self, data, laplace_zeta):
        def expected_sample_gamma(size):
            return np.random.gamma(shape=3, size=size)
        def expected_sample_uniform(size):
            return np.random.uniform(size=size)
        data_norm = jnp.linalg.norm(data, axis=-1)
        x, y, z = data.T
        theta = jnp.arccos(z / data_norm)
        sin_phi_sign = y / jnp.sin(theta)

        phi_between_half_pi = jnp.arctan(y / x)
        phi =(
            phi_between_half_pi
            + (phi_between_half_pi < 0) * (sin_phi_sign > 0) * jnp.pi
            - (phi_between_half_pi > 0) * (sin_phi_sign < 0) * jnp.pi
        )

        gamma_vars = data_norm * 2 * laplace_zeta
        uniform_vars = (phi + jnp.pi) / 2 / jnp.pi

        test_result_norm = sp.stats.kstest(expected_sample_gamma, gamma_vars)
        test_result_phi = sp.stats.kstest(expected_sample_uniform, uniform_vars)
        self.assertGreater(test_result_norm.pvalue, 0.1)
        self.assertGreater(test_result_phi.pvalue, 0.1)


        correct_l = 0
        total_l = 0
        for l in np.arange(0, np.pi, 0.01):
            n = np.sum(theta < l)
            if np.isclose(n / len(theta), (1 - np.cos(l)) / 2, rtol=0.05):
                correct_l += 1
            total_l += 1
        self.assertGreater(correct_l / total_l, 0.95)


    def test_sample_gamma(self):

        laplace_zeta = 10
        num_data = 1000000
        seed = 42
        key = jax.random.PRNGKey(seed)

        all_keys = jax.random.split(key, num_data)
        vmapped_sample_func = jax.vmap(sample_gamma, in_axes=(0, None))
        data = vmapped_sample_func(all_keys, laplace_zeta)
        self.do_test_gamma(data, laplace_zeta)

    def test_calc_G_log(self):
        updated_position = jnp.zeros((3, ))
        nearest_nucleus = jnp.array([1.0, 0.0, 0.0])
        drifted_position = jnp.array([2.0, 0.0, 0.0])
        nucleus_overshoot_prob = 0.1

        time_step = 0.01
        laplace_zeta = 1 / jnp.sqrt(time_step)

        G_log = calc_G_log(updated_position, nearest_nucleus, drifted_position,
                           nucleus_overshoot_prob, laplace_zeta, time_step)
        expected_G_log = jnp.log(
            (1 - nucleus_overshoot_prob) * (2 * jnp.pi * time_step) ** (-1.5) * jnp.exp(-(jnp.linalg.norm(updated_position - drifted_position) ** 2) / 2 / time_step)
            + nucleus_overshoot_prob * laplace_zeta ** 3 / jnp.pi * jnp.exp(-2 * laplace_zeta * jnp.linalg.norm(updated_position - nearest_nucleus))
        )
        self.assertAlmostEqual(G_log, expected_G_log)

if __name__ == '__main__':
  absltest.main()
