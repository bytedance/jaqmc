# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from jax_cosmo.scipy import interpolate
import numpy as np

def get_forward_laplacian_for_kinetic_ph(f, raw_ph_data, ph_atom_pos, rv_type='spline'):
    if rv_type == 'spline':
        return get_forward_laplacian_for_kinetic_ph_spline(f, raw_ph_data, ph_atom_pos)
    elif rv_type == 'linear':
        return get_forward_laplacian_for_kinetic_ph_linear(f, raw_ph_data, ph_atom_pos)
    else:
        raise NotImplementedError(f'rv_type {rv_type} not supported yet')

def get_forward_laplacian_for_kinetic_ph_linear(f, raw_ph_data, ph_atom_pos):
    from lapjax import LapTuple, TupType
    from lapjax.numpy import matmul as lapjax_matmul

    rx = jnp.array(np.linspace(0, 10.0, 10001))
    def rV(r, arr):
        return jnp.interp(r, rx, jnp.asarray(arr))

    def prepare_ph_for_all_atoms():
        all_loc_data = []
        all_l2_data = []
        all_atom_pos = []
        for atom, _atom_pos in ph_atom_pos:
            loc_data, l2_data = raw_ph_data[atom]
            all_loc_data.append(loc_data)
            all_l2_data.append(l2_data)
            all_atom_pos.append(_atom_pos)
        return (
            jnp.array(all_loc_data),
            jnp.array(all_l2_data),
            jnp.array(all_atom_pos))
    ph_data = prepare_ph_for_all_atoms()

    @jax.vmap
    def get_eff_mass(x):
        def for_scan_f(carry, input):
            loc_data, l2_data, atom_pos = input
            r = jnp.linalg.norm(x - atom_pos)
            rv_l2_val = rV(r, l2_data)
            diag = rv_l2_val * r
            mass = -rv_l2_val * jnp.outer(x - atom_pos, x - atom_pos) / r + jnp.identity(3) * diag
            return mass + carry, None

        mass, _ = jax.lax.scan(for_scan_f, jnp.zeros([3, 3]), ph_data)
        mass += 0.5 * jnp.identity(3)
        return mass

    def kinetic_ph(params, data):
        f_closure = lambda x: f(params, x)
        data_per_elec = data.reshape(-1, 3)

        def second_order():
            mass = get_eff_mass(data_per_elec)
            dim = 3
            L = jnp.linalg.cholesky(mass).transpose(0, 2, 1)
            input_laptuple = LapTuple(data, is_input=True)
            input_laptuple: LapTuple = lapjax_matmul(input_laptuple.reshape(-1, dim), jnp.eye(dim))
            input_laptuple.grad = jnp.matmul(L,input_laptuple.grad.transpose(1, 0, 2)).transpose(1, 0, 2)
            output = f(params,input_laptuple)
            return -output.get(TupType.LAP) - jnp.sum(output.get(TupType.GRAD) ** 2)

        def first_order():
            @jax.vmap
            def v_dot(grad, x):
                def for_scan_f(carry, input):
                    loc_data, l2_data, atom_pos = input
                    rel_pos = x - atom_pos
                    r = jnp.linalg.norm(rel_pos)
                    return carry + rV(r, l2_data) * 2 / r * rel_pos, None

                output, _ = jax.lax.scan(for_scan_f, jnp.zeros(3), ph_data)
                return jnp.dot(grad, output)

            grad = jax.grad(f_closure)(data)
            grad_blocks = grad.reshape(-1, 3)
            return jnp.sum(v_dot(grad_blocks, data_per_elec))

        def zero_order():
            @jax.vmap
            def f(x):
                def for_scan_f(carry, input):
                    loc_data, l2_data, atom_pos = input
                    r = jnp.linalg.norm(x - atom_pos)
                    return carry + rV(r, loc_data) / r, None
                output, _ = jax.lax.scan(for_scan_f, 0.0, ph_data)
                return output

            return jnp.sum(f(data_per_elec))
        return (
                second_order()
                + first_order() + zero_order())
    return kinetic_ph

def get_forward_laplacian_for_kinetic_ph_spline(f, raw_ph_data, ph_atom_pos):
    from lapjax import LapTuple, TupType
    from lapjax.numpy import matmul as lapjax_matmul

    def prepare_rV_cubic_spline():
        rx = jnp.array(np.linspace(0, 10.0, 10001))
        rV_dict = {}
        i_atom = 0
        for atom, (loc_data, l2_data) in raw_ph_data.items():
            rV_dict[(i_atom, 0)] = interpolate.InterpolatedUnivariateSpline(rx, loc_data, k=3)
            rV_dict[(i_atom, 1)] = interpolate.InterpolatedUnivariateSpline(rx, l2_data, k=3)
            i_atom += 1

        def rV(r, i_atom, mode=0):
            # mode=0 for loc_data, mode=1 for l2_data
            return rV_dict[(i_atom, mode)](r)

        return rV

    rV = prepare_rV_cubic_spline()

    def prepare_ph_for_all_atoms():
        res = []
        for atom, _atom_pos in ph_atom_pos:
            res.append((jnp.array(_atom_pos), list(raw_ph_data).index(atom)))
        return res
    ph_data = prepare_ph_for_all_atoms()

    @jax.vmap
    def get_eff_mass(x):
        mass = jnp.zeros([3, 3])
        for atom_pos, i_atom in ph_data:
            r = jnp.linalg.norm(x - atom_pos)
            rv_l2_val = rV(r, i_atom, mode=1)
            diag = rv_l2_val * r
            mass += -rv_l2_val * jnp.outer(x - atom_pos, x - atom_pos) / r + jnp.identity(3) * diag
        mass += 0.5 * jnp.identity(3)
        return mass

    def kinetic_ph(params, data):
        f_closure = lambda x: f(params, x)
        data_per_elec = data.reshape(-1, 3)

        def second_order():
            mass = get_eff_mass(data_per_elec)
            dim = 3
            L = jnp.linalg.cholesky(mass).transpose(0, 2, 1)
            input_laptuple = LapTuple(data, is_input=True)
            input_laptuple: LapTuple = lapjax_matmul(input_laptuple.reshape(-1, dim), jnp.eye(dim))
            input_laptuple.grad = jnp.matmul(L,input_laptuple.grad.transpose(1, 0, 2)).transpose(1, 0, 2)
            output = f(params,input_laptuple)
            return -output.get(TupType.LAP) - jnp.sum(output.get(TupType.GRAD) ** 2)

        def first_order():
            @jax.vmap
            def v_dot(grad, x):
                carry = jnp.zeros(3)
                for atom_pos, i_atom in ph_data:
                    rel_pos = x - atom_pos
                    r = jnp.linalg.norm(rel_pos)
                    carry += rV(r, i_atom, mode=1) * 2 / r * rel_pos
                return jnp.dot(grad, carry)

            grad = jax.grad(f_closure)(data)
            grad_blocks = grad.reshape(-1, 3)
            return jnp.sum(v_dot(grad_blocks, data_per_elec))

        def zero_order():
            @jax.vmap
            def f(x):
                carry = 0.0
                for atom_pos, i_atom in ph_data:
                    r = jnp.linalg.norm(x - atom_pos)
                    carry += rV(r, i_atom, mode=0) / r
                return carry

            return jnp.sum(f(data_per_elec))
        return (
                second_order()
                + first_order()
                + zero_order())
    return kinetic_ph
