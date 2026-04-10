# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from .quadrature import Quadrature
from .special import legendre_list

from typing import Tuple

def numerical_integral_for_loop(psi, rea_v, rea, walkers, max_l, key, quadrature: Quadrature, ecp_range: Tuple):
    '''
    ref: Nonlocal pseudopotentials and diffusion Monte Carlo, equation 28

    inputs:
        psi: wave function that psi(walkers) returns a complex number
        rea_v: vector between electron and atoms
        rea: distance between electron and atoms
        walkers: shape (n_electron, 3)
        max_l: shape(l_number,) values of l to evaluate
        key: random number
        quadrature: A quadrature object to do numerical integration.
        ecp_range: (n_up_start, n_up_end, n_down_start, n_down_end)
    returns:
        value of the integral \int (2l+1) * P_l(cos theta) psi(r1,..,ri,..)
        shape (n_electron, l_number)
    '''

    n_electron = walkers.shape[0]
    ls = list(range(max_l-1))
    res = jnp.zeros((n_electron, len(ls)))
    res_den = psi(walkers)
    normal_walkers = rea_v / rea[:, None]
    psi_vec = jax.vmap(psi, in_axes=0)

    def psi_r(i, x):
        coords = x.reshape(-1,3) * rea[i] + walkers[i] - rea_v[i]
        new_walkers = jnp.tile(walkers, (coords.shape[0],) + (1, 1))
        new_walkers = new_walkers.at[:,i,:].set(coords)
        res_num = psi_vec(new_walkers)
        res = res_num[0] * res_den[0] * jnp.exp(res_num[1] - res_den[1])
        return res.reshape(x.shape[:-1])
    npts = quadrature.discrete_pts(key, N=n_electron)

    Pl_ = lambda x: legendre_list(x, ls)
    def Pl(i, x):
        tmp = Pl_(jnp.matmul(x, normal_walkers[i, :]))  ## Pl(cos(\theta))
        return tmp
    def integral(i, res):
        result = quadrature.integral_value(Pl(i,npts[i]),psi_r(i,npts[i])) * (2 * jnp.array(ls) + 1)
        res = res.at[i,:].set(result)
        return res
    n_up_start, n_up_end, n_down_start, n_down_end = ecp_range
    res = jax.lax.fori_loop(n_up_start, n_up_end, integral, res)
    res = jax.lax.fori_loop(n_down_start, n_down_end, integral, res)

    return res

def numerical_integral_optim_for_loop(f_modify, f_memory, f_memory_update,
                                      rea_v, rea, walkers, max_l, key, quadrature: Quadrature,
                                      ecp_range: Tuple):
    '''
    ref: Nonlocal pseudopotentials and diffusion Monte Carlo, equation 28

    inputs:
        psi: wave function that psi(walkers) returns a complex number
        rea_v: vector between electron and atoms
        rea: distance between electron and atoms
        walkers: shape (n_electron, 3)
        max_l: shape(l_number,) values of l to evaluate
        key: random number
        quadrature: A quadrature object to do numerical integration.
        ecp_range: (n_up_start, n_up_end, n_down_start, n_down_end)
    returns:
        value of the integral \int (2l+1) * P_l(cos theta) psi(r1,..,ri,..)
        shape (n_electron, l_number)
    '''

    n_electron = walkers.shape[0]
    ls = list(range(max_l-1))
    res = jnp.zeros((n_electron, len(ls)))
    memory = f_memory(walkers)
    res_den = f_modify(memory, walkers)
    normal_walkers = rea_v / rea[:, None]
    f_memory_update_vec = jax.vmap(f_memory_update, in_axes=(None, 0, None))
    f_modify_vec = jax.vmap(f_modify, in_axes=(0, 0), out_axes=(0))

    def psi_r(i, x):
        coords = x.reshape(-1,3) * rea[i] + walkers[i] - rea_v[i]
        new_walkers = jnp.tile(walkers, (coords.shape[0],) + (1, 1))
        new_walkers = new_walkers.at[:,i,:].set(coords)
        memory_vec = f_memory_update_vec(memory, new_walkers, i)
        res_num = f_modify_vec(memory_vec, new_walkers)
        res = res_num[0] * res_den[0] * jnp.exp(res_num[1] - res_den[1])
        return res.reshape(x.shape[:-1])
    npts = quadrature.discrete_pts(key, N=n_electron)

    Pl_ = lambda x: legendre_list(x, ls)
    def Pl(i, x):
        tmp = Pl_(jnp.matmul(x, normal_walkers[i, :]))  ## Pl(cos(\theta))
        return tmp
    def integral(i, res):
        result = quadrature.integral_value(Pl(i,npts[i]),psi_r(i,npts[i])) * (2 * jnp.array(ls) + 1)
        res = res.at[i,:].set(result)
        return res
    n_up_start, n_up_end, n_down_start, n_down_end = ecp_range
    res = jax.lax.fori_loop(n_up_start, n_up_end, integral, res)
    res = jax.lax.fori_loop(n_down_start, n_down_end, integral, res)

    return res


def numerical_integral(psi, rea_v, rea, walkers, max_l, key,
                        quadrature: Quadrature, ecp_range: Tuple):
    '''
    ref: Nonlocal pseudopotentials and diffusion Monte Carlo, equation 28

    inputs:
        psi: wave function that psi(walkers) returns a complex number
        rea_v: vector between electron and atoms
        rea: distance between electron and atoms
        walkers: shape (n_electron, 3)
        max_l: shape(l_number,) values of l to evaluate
        key: random number
        quadrature: A quadrature object to do numerical integration.
    returns:
        value of the integral \int (2l+1) * P_l(cos theta) psi(r1,..,ri,..)
        shape (n_electron, l_number)
    '''

    numerical_integral_exact_closure = lambda rv, r : \
                numerical_integral_for_loop(psi, rv, r, walkers, max_l, key, quadrature, ecp_range)
    numerical_integral_exact_vmap = jax.vmap(numerical_integral_exact_closure, in_axes=(1,1), out_axes=(0))
    res = numerical_integral_exact_vmap(rea_v, rea)
    return res.transpose(1,0,2)

def numerical_integral_optim(f_modify, f_memory, f_memory_update, rea_v, rea,
                            walkers, max_l, key, quadrature: Quadrature, ecp_range: Tuple):
    '''
    ref: Nonlocal pseudopotentials and diffusion Monte Carlo, equation 28

    inputs:
        psi: wave function that psi(walkers) returns a complex number
        rea_v: vector between electron and atoms
        rea: distance between electron and atoms
        walkers: shape (n_electron, 3)
        max_l: shape(l_number,) values of l to evaluate
        key: random number
        quadrature: A quadrature object to do numerical integration.
    returns:
        value of the integral \int (2l+1) * P_l(cos theta) psi(r1,..,ri,..)
        shape (n_electron, l_number)
    '''

    numerical_integral_exact_closure = lambda rv, r : \
                numerical_integral_optim_for_loop(f_modify, f_memory, f_memory_update, rv, r, walkers, max_l, key, quadrature, ecp_range)
    numerical_integral_exact_vmap = jax.vmap(numerical_integral_exact_closure, in_axes=(1,1), out_axes=(0))
    res = numerical_integral_exact_vmap(rea_v, rea)
    return res.transpose(1,0,2)
