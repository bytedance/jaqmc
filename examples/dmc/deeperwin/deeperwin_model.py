# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Largely directly copied from DeepErwin repo with minor modification to match
DMC interface.
'''

from deeperwin.model import Wavefunction, init_model_fixed_params
from deeperwin.configuration import ModelConfig, PhysicalConfig
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

def evaluate_sum_of_determinants_with_sign(mo_matrix_up, mo_matrix_dn, use_full_det):
    LOG_EPSILON = 1e-8

    if use_full_det:
        mo_matrix = jnp.concatenate([mo_matrix_up, mo_matrix_dn], axis=-2)
        sign_total, log_total = jnp.linalg.slogdet(mo_matrix)
    else:
        sign_up, log_up = jnp.linalg.slogdet(mo_matrix_up)
        sign_dn, log_dn = jnp.linalg.slogdet(mo_matrix_dn)
        log_total = log_up + log_dn
        sign_total = sign_up * sign_dn
    log_shift = jnp.max(log_total, axis=-1, keepdims=True)
    psi = jnp.exp(log_total - log_shift) * sign_total
    psi = jnp.sum(psi, axis=-1)  # sum over determinants
    sign = jnp.sign(psi)
    log_psi_sqr = 2 * (jnp.log(jnp.abs(psi) + LOG_EPSILON) + jnp.squeeze(log_shift, -1))
    return sign, log_psi_sqr

class UpdatedWavefunction(Wavefunction):
    def __init__(self, config: ModelConfig, phys_config: PhysicalConfig, name="wf"):
        super().__init__(name=name, config=config, phys_config=phys_config)

    def __call__(self, r, R, Z, fixed_params=None):
        fixed_params = fixed_params or {}
        diff_dist, features = self._calculate_features(r, R, Z, fixed_params.get('input'))
        embeddings = self._calculate_embedding(features)
        mo_up, mo_dn = self._calculate_orbitals(diff_dist, embeddings, fixed_params.get('orbitals'))

        # This is the main change
        psi_sign, log_psi_sqr = evaluate_sum_of_determinants_with_sign(mo_up, mo_dn, self.config.orbitals.use_full_det)

        # Jastrow factor to the total wavefunction
        if self.config.jastrow:
            log_psi_sqr += self._calculate_jastrow(embeddings)

        # Electron-electron-cusps
        if self.config.use_el_el_cusp_correction:
            log_psi_sqr += self._el_el_cusp(diff_dist.dist_el_el)
        return psi_sign, log_psi_sqr

def build_log_psi_squared_with_sign(config: ModelConfig, phys_config: PhysicalConfig, fixed_params=None):
    # Initialize fixed model parameters
    fixed_params = fixed_params or init_model_fixed_params(config, phys_config)

    # Build model
    model = hk.multi_transform(lambda: UpdatedWavefunction(config, phys_config).init_for_multitransform())

    # Initialized trainable parameters using a dummy batch
    n_el, _, R, Z = phys_config.get_basic_params()
    r = np.random.normal(size=[1, n_el, 3])
    rng = jax.random.PRNGKey(np.random.randint(2**31))
    params = model.init(rng, r, R, Z, fixed_params)

    # Remove rng-argument (replace by None) and move parameters to back of function
    log_psi_sqr_with_sign = lambda params, *batch: model.apply[0](params, None, *batch)
    orbitals = lambda params, *batch: model.apply[1](params, None, *batch)

    return log_psi_sqr_with_sign, orbitals, params, fixed_params
