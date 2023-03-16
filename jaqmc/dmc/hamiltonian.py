# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.


"""Evaluating the Hamiltonian on a wavefunction."""

from typing import Tuple

import jax
from jax import lax
import jax.numpy as jnp

from .utils import agg_mean


def local_kinetic_energy(f, partition_num=0):
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: log wave function. Takes in electronic configuration and output log of wavefunction.
    partition_num: 0: fori_loop implementation
                   1: Hessian implementation
                   other positive integer: Split the laplacian to multiple trunks and
                                           calculate accordingly.

  Returns:
    Callable with signature lapl(data), which evaluates the local
    kinetic energy, -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| +
    (\nabla log|f|)^2).
  """
  vjvp = jax.vmap(jax.jvp, in_axes=(None, None, 0))

  def _lapl_over_f(x):
    n = x.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.grad(f)
    # Use Hessian
    if partition_num == 1:
        g = grad_f(x)
        hess = jax.hessian(f)(x)
        return -0.5 * (jnp.trace(hess) + jnp.sum(g ** 2))

    # Original implementation
    if partition_num == 0:
        def _body_fun(i, val):
          primal, tangent = jax.jvp(grad_f, (x,), (eye[i],))
          return val + primal[i]**2 + tangent[i]
        return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)

    # vjvp implementation
    assert n % partition_num == 0, f'partition_num {partition_num} does not divide the dimension {n}'
    eyes = jnp.asarray(jnp.array_split(eye, partition_num))

    def _body_fun(val, e):
        primal, tangent = vjvp(grad_f, (x,), (e,))
        return val, (primal, tangent)

    _, (primal, tangent) = lax.scan(_body_fun, None, eyes)
    primal = primal.reshape((-1, primal.shape[-1]))
    tangent = tangent.reshape((-1, tangent.shape[-1]))
    return -0.5 * (jnp.sum(jnp.diagonal(primal) ** 2) + jnp.trace(tangent))

  return _lapl_over_f


def potential_energy(r_ae, r_ee, atoms, charges):
  """Returns the potential energy for this electron configuration.

  Args:
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
  """
  v_ee = jnp.sum(jnp.triu(1 / r_ee[..., 0], k=1))
  v_ae = -jnp.sum(charges / r_ae[..., 0])  # pylint: disable=invalid-unary-operand-type
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  v_aa = jnp.sum(
      jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))
  return v_ee + v_ae + v_aa


def get_dist(
    x: jnp.ndarray,
    atoms: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Calculate distance between electron and atomic positions.

  Args:
    x: electron positions. Shape (nelectrons * 3,).
    atoms: atom positions. Shape (natoms, 3).

  Returns:
    r_ae, r_ee pair, where:
      r_ae: atom-electron distance. Shape (nelectron, natom, 1).
      r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
    The diagonal terms in r_ee are masked out such that the gradients of these
    terms are also zero.
  """
  ndim = 3
  ae = jnp.reshape(x, [-1, 1, ndim]) - atoms[None, ...]
  ee = jnp.reshape(x, [1, -1, ndim]) - jnp.reshape(x, [-1, 1, ndim])

  r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
  # Avoid computing the norm of zero, as is has undefined grad
  n = ee.shape[0]
  r_ee = (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))

  return r_ae, r_ee[..., None]


def local_energy(f, atoms, charges, el_partition_num=0):
  """Creates function to evaluate the local energy.

  Args:
    f: Callable with signature f(data) which returns the log magnitude
      of the wavefunction given configurations data.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
    partition_num: 0: fori_loop implementation
                   1: Hessian implementation
                   other positive integer: Split the laplacian to multiple trunks and
                                           calculate accordingly.

  Returns:
    Callable with signature e_l(data) which evaluates the local energy
    of the wavefunction given single MCMC configuration in data.
  """
  ke = local_kinetic_energy(f, el_partition_num)

  def _e_l(x):
    """Returns the total energy.

    Args:
      x: MCMC configuration.
    """
    r_ae, r_ee = get_dist(x, atoms)
    potential = potential_energy(r_ae, r_ee, atoms, charges)
    kinetic = ke(x)
    return potential + kinetic

  return _e_l

def make_calc_energy_func(el_fun, clip_pair=None):
    '''
    A factory for averaged energy calculation using local_energy func `el_fun` on a batch of walkers.
    '''
    # position and mask are vectorized, not params
    vmap_in_axes = (0, 0)
    pmap_in_axes = (0, 0)

    def local_energy_func_with_mask(position, mask):
        return jax.lax.cond(
            mask,
            el_fun,
            lambda _: 0.0,
            position)

    num_device = jax.local_device_count()
    pmaped_energy_func = jax.pmap(
        jax.vmap(local_energy_func_with_mask, in_axes=vmap_in_axes),
        in_axes=pmap_in_axes)

    def calc_energy(flatten_position):
        num_walkers, walker_dim = flatten_position.shape
        if num_walkers % num_device == 0:
            target_num_walkers = num_walkers
        else:
            target_num_walkers = num_walkers + num_device - (num_walkers % num_device)
        to_pad_num = target_num_walkers - num_walkers
        mask = jnp.pad(
            jnp.ones(num_walkers),
            ((0, to_pad_num),),
            constant_values=0).reshape((num_device, -1))
        position = jnp.pad(
            flatten_position,
            ((0, to_pad_num), (0, 0)),
            constant_values=0).reshape((num_device, -1, walker_dim))
        _local_energy = pmaped_energy_func(position, mask)
        return calc_masked_energy(_local_energy, mask, clip_pair=clip_pair)

    return calc_energy

def calc_masked_energy(local_energy, mask, clip_pair=None):
    if clip_pair is None:
        return agg_mean(local_energy, mask)
    clip_min, clip_max = clip_pair
    clipped_energy = jnp.clip(local_energy, clip_min, clip_max)
    return agg_mean(clipped_energy, mask)
