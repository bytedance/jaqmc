# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Drifting and diffusion for DMC walkers.
'''

import jax
import jax.numpy as jnp

def single_electron_wrapper(position, velocity, nuclei, charges,
                            time_step, key, updated_position=None):
    _, _, nearest_index = to_nearest_nucleus(position, nuclei)
    nearest_nucleus = nuclei[nearest_index]
    nearest_charge = charges[nearest_index]
    laplace_zeta = jnp.sqrt(nearest_charge ** 2 + 1 / time_step)

    clipped_velocity = clip_velocity_helper(
        velocity=velocity,
        position=position,
        nearest_nucleus=nearest_nucleus,
        nearest_charge=nearest_charge,
        time_step=time_step)
    drifted_position = do_drift(
        position=position,
        nearest_nucleus=nearest_nucleus,
        clipped_velocity=clipped_velocity,
        time_step=time_step)
    nucleus_overshoot_prob = calc_nucleus_overshoot_prob_helper(
        position=position,
        nearest_nucleus=nearest_nucleus,
        clipped_velocity=clipped_velocity,
        time_step=time_step)
    if updated_position is None:
        updated_position, delta_R_vec = do_diffusion(
            drifted_position=drifted_position,
            nearest_nucleus=nearest_nucleus,
            nucleus_overshoot_prob=nucleus_overshoot_prob,
            key=key,
            time_step=time_step,
            laplace_zeta=laplace_zeta)
    else:
        delta_R_vec = None
    G_log = calc_G_log(
        updated_position=updated_position,
        nearest_nucleus=nearest_nucleus,
        drifted_position=drifted_position,
        nucleus_overshoot_prob=nucleus_overshoot_prob,
        laplace_zeta=laplace_zeta,
        time_step=time_step)
    return updated_position, clipped_velocity, G_log, delta_R_vec

def from_orig_config_wrapper(position, velocity, nuclei, charges,
                             time_step, key):
    single_electron_func = lambda se_pos, se_velocity, se_key: single_electron_wrapper(
        se_pos, se_velocity,
        nuclei, charges,
        time_step, se_key)

    vmapped_func = jax.vmap(single_electron_func)
    reshaped_position = position.reshape((-1, 3))
    reshaped_velocity = velocity.reshape((-1, 3))
    key = jax.random.split(key, reshaped_position.shape[0])
    updated_position, clipped_velocity, G_log, delta_R_vec = vmapped_func(reshaped_position, reshaped_velocity, key)
    delta_R = jnp.linalg.norm(delta_R_vec)
    return (updated_position.reshape((-1, )),
            clipped_velocity.reshape((-1, )),
            jnp.sum(G_log),
            delta_R)

def from_updated_config_wrapper(orig_position, updated_position, updated_velocity,
                                nuclei, charges,
                                time_step, key):
    single_electron_func = lambda se_pos, se_updated_pos, se_velocity, se_key: single_electron_wrapper(
        se_pos, se_velocity,
        nuclei, charges,
        time_step, se_key,
        updated_position=se_updated_pos)

    vmapped_func = jax.vmap(single_electron_func)
    reshaped_position = orig_position.reshape((-1, 3))
    reshaped_updated_position = updated_position.reshape((-1, 3))
    reshaped_updated_velocity = updated_velocity.reshape((-1, 3))
    key = jax.random.split(key, reshaped_position.shape[0])

    # The mismatch of argument here is as intended: we are treating the updated
    # position/velocity as original position/velocity.
    updated_position, clipped_velocity, G_log, _ = vmapped_func(
        reshaped_updated_position,
        reshaped_position,
        reshaped_updated_velocity, key)

    return (updated_position.reshape((-1, )),
            clipped_velocity.reshape((-1, )),
            jnp.sum(G_log))

def from_orig_config_wrapper_ebye(
    position, velocity, nuclei, charges, signed_psi, v_func, time_step, key
):
    single_electron_func = (
        lambda se_pos, se_updated_pos, se_velocity, se_key: single_electron_wrapper(
            se_pos,
            se_velocity,
            nuclei,
            charges,
            time_step,
            se_key,
            updated_position=se_updated_pos,
        )
    )
    # To get backward green function value, offer the original position as se_updated_pos.
    # To get updated electron position, clipped velocity, forward green function value
    # from original position, just let se_updated_pos be None.

    def helper(index, values):
        (
            reshaped_position,
            reshaped_velocity,
            sum_p,
            sum_p_times_square_delta_R,
            sum_square_delta_R,
            key,
        ) = values
        updated_position = reshaped_position # for proposal

        key, subkey = jax.random.split(key)
        (
            updated_position_i,
            _,
            G_forward,
            delta_R_vec,
        ) = single_electron_func(
            reshaped_position[index], None, reshaped_velocity[index], subkey
        )
        updated_position = updated_position.at[index].set(updated_position_i)
        square_delta_R = jnp.sum(delta_R_vec ** 2)
        sum_square_delta_R += square_delta_R

        updated_velocity = v_func(updated_position.reshape((-1, ))).reshape((-1, 3))

        key, subkey = jax.random.split(key)
        (
            _,
            _,
            G_backward,
            _,
        ) = single_electron_func(
            updated_position[index],
            reshaped_position[index],
            updated_velocity[index],
            subkey,
        )

        sign_orig, psi_orig = signed_psi(reshaped_position.reshape((-1, )))
        sign_updated, psi_updated = signed_psi(updated_position.reshape((-1, )))

        logp_accept_elec = (
        2 * (psi_updated - psi_orig)
        + G_backward - G_forward
        )
        p_accept_elec = jnp.clip(jnp.exp(logp_accept_elec), 0.0, 1.0) * jnp.asarray(
            sign_orig * sign_updated > 0, dtype=jnp.float32
        ) # fix the node
        sum_p += p_accept_elec
        sum_p_times_square_delta_R += p_accept_elec * square_delta_R

        key, subkey = jax.random.split(key)
        cond = jax.random.uniform(subkey) < p_accept_elec
        reshaped_position = jnp.where(cond, updated_position, reshaped_position)
        reshaped_velocity = jnp.where(cond, updated_velocity, reshaped_velocity)

        return (
            reshaped_position,
            reshaped_velocity,
            sum_p,
            sum_p_times_square_delta_R,
            sum_square_delta_R,
            key,
        )

    reshaped_position = position.reshape((-1, 3))
    reshaped_velocity = velocity.reshape((-1, 3))
    clipped_velocity = jnp.ones(reshaped_velocity.shape)

    # clip the original velocity
    # For convenience, we directly use vmapped single_electron_func
    # to clip the velocity, but note that there are some extra
    # unnecessary caculations.
    vmapped_func = jax.vmap(single_electron_func)
    key, subkey = jax.random.split(key)
    subkey = jax.random.split(subkey, reshaped_position.shape[0])
    _, clipped_velocity, _, _ = vmapped_func(
        reshaped_position,
        None, reshaped_velocity, subkey)
    
    # Moving of one single electron may change the global velocity,
    # so the velocity clipping should be done outside the loop
    # instead of clipping one by one in the loop.
    (
        reshaped_position,
        reshaped_velocity,
        sum_p,
        sum_p_times_square_delta_R,
        sum_square_delta_R,
        key,
    ) = jax.lax.fori_loop(
        0,
        reshaped_velocity.shape[0],
        helper,
        (
            reshaped_position,
            reshaped_velocity,
            0, 0, 0,
            key,
        )
    )

    # clip the final updated_velocity
    subkey = jax.random.split(key, reshaped_position.shape[0])
    _, clipped_updated_velocity, _, _ = vmapped_func(
        reshaped_position,
        position.reshape((-1, 3)),
        reshaped_velocity, subkey)

    mean_p = sum_p/reshaped_velocity.shape[0]
    effective_time_step = time_step * sum_p_times_square_delta_R / sum_square_delta_R
    return (
        reshaped_position.reshape((-1,)), # updated position
        clipped_velocity.reshape((-1,)), # clipped orig velocity
        reshaped_velocity.reshape((-1,)), # updated velocity
        clipped_updated_velocity.reshape((-1,)), # clipped updated velocity
        mean_p,
        effective_time_step,
    )

def do_drift(position, nearest_nucleus, clipped_velocity, time_step):
    vec_to_nearest_nucleus = position - nearest_nucleus
    nearest_dist = jnp.linalg.norm(vec_to_nearest_nucleus)
    unit_vec_to_nearest_nucleus = vec_to_nearest_nucleus / nearest_dist

    v_z = jnp.dot(clipped_velocity, unit_vec_to_nearest_nucleus)
    v_rho_vec = clipped_velocity - v_z * unit_vec_to_nearest_nucleus
    v_rho = jnp.linalg.norm(v_rho_vec)
    unit_v_rho_vec = v_rho_vec / v_rho
    z_double_prime = jnp.clip(nearest_dist + v_z * time_step, 0, jnp.inf)
    rho_double_prime = 2 * v_rho * time_step * z_double_prime / (nearest_dist + z_double_prime)
    return (nearest_nucleus
            + z_double_prime * unit_vec_to_nearest_nucleus
            + rho_double_prime * unit_v_rho_vec)

def do_diffusion(drifted_position,
                 nearest_nucleus, nucleus_overshoot_prob,
                 key, time_step, laplace_zeta):
    r_prime = drifted_position + jax.random.normal(key, (3, )) * jnp.sqrt(time_step)
    p_tilde = 1 - nucleus_overshoot_prob

    key, subkey = jax.random.split(key)
    uniform_var = jax.random.uniform(subkey)
    drifted_position, delta_R_vec = jax.lax.cond(
        uniform_var < p_tilde,
        lambda _: (drifted_position, jax.random.normal(key, (3, )) * jnp.sqrt(time_step)),
        lambda _: (nearest_nucleus, sample_gamma(key, laplace_zeta)),
        operand=None
    )
    r_prime = drifted_position + delta_R_vec
    return r_prime, delta_R_vec

def sample_gamma(key, laplace_zeta):
    key1, key2, key3 = jax.random.split(key, num=3)
    # 'a == 3' corresponding to x^2 * e^(-x)
    norm = jax.random.gamma(a=3, key=key1) * 0.5 / laplace_zeta
    phi = jax.random.uniform(key=key2, minval=0, maxval=2 * jnp.pi)
    theta = jnp.arccos(1 - 2 * jax.random.uniform(key=key3))
    return jnp.array(
        [norm * jnp.cos(phi) * jnp.sin(theta),
         norm * jnp.sin(phi) * jnp.sin(theta),
         norm * jnp.cos(theta)])

def calc_nucleus_overshoot_prob_helper(position, nearest_nucleus, clipped_velocity,
                                       time_step):
    vec_to_nearest_nucleus = position - nearest_nucleus
    nearest_dist = jnp.linalg.norm(vec_to_nearest_nucleus)
    v_z = jnp.dot(clipped_velocity, vec_to_nearest_nucleus / nearest_dist)
    q_tilde = 0.5 * jax.lax.erfc((nearest_dist + v_z * time_step) / jnp.sqrt(2 * time_step))
    return q_tilde

def calc_G_log(updated_position, nearest_nucleus, drifted_position,
               nucleus_overshoot_prob, laplace_zeta, time_step):
    def g1(xi):
        log = -1.5 * jnp.log(2 * jnp.pi * time_step) - 0.5 * (xi ** 2).sum() / time_step
        return jnp.exp(log)
    def g2(xi):
        log = 3 * jnp.log(laplace_zeta) - 2 * laplace_zeta * jnp.linalg.norm(xi)
        return jnp.exp(log) / jnp.pi

    p_tilde = 1 - nucleus_overshoot_prob
    non_overshoot_part = p_tilde * g1(updated_position - drifted_position)
    overshoot_part = (1 - p_tilde) * g2(updated_position - nearest_nucleus)

    G = non_overshoot_part + overshoot_part
    return jnp.log(G)

def to_nearest_nucleus(position, nuclei, normalized=True):
    difference = position - nuclei
    norm = jnp.expand_dims(jnp.linalg.norm(difference, axis=1), axis=-1)
    if normalized:
        difference /= norm
    nearest_index = jnp.argmin(norm)
    return difference[nearest_index], norm[nearest_index], nearest_index

def calc_per_electron_weight_control_helper(velocity, position, nearest_nucleus, nearest_charge):
    normed_velocity = velocity / jnp.linalg.norm(velocity)
    vec_to_nearest_nucleus = position - nearest_nucleus
    nearest_dist = jnp.linalg.norm(vec_to_nearest_nucleus)
    unit_vec_to_nearest_nucleus = vec_to_nearest_nucleus / nearest_dist
    return (0.5 * (1 + jnp.dot(normed_velocity, unit_vec_to_nearest_nucleus))
            + 0.1 /  (4 / ((nearest_charge * nearest_dist) ** 2) + 1))

def clip_velocity_helper(velocity, position, nearest_nucleus, nearest_charge, time_step):
    # In float32, for x < 2e-4, then the floating numerical error start to show up
    def calc_clip_factor(x):
        return (-1 + jnp.sqrt(1 + 2 * x)) / x
    clip_threshold = 2e-4

    weight_control = calc_per_electron_weight_control_helper(
        velocity=velocity,
        position=position,
        nearest_nucleus=nearest_nucleus,
        nearest_charge=nearest_charge)
    v_norm_squared = jnp.sum(jnp.square(velocity))

    value = weight_control * v_norm_squared * time_step
    clip_factor = jax.lax.cond(
        value < clip_threshold,
        lambda _: 1.0,
        calc_clip_factor,
        operand=value)
    return clip_factor * velocity
