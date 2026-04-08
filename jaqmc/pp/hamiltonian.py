# Copyright 2020 DeepMind Technologies Limited.
# Copyright 2025 Bytedance Ltd. and/or its affiliate
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

"""Evaluating the Hamiltonian on a wavefunction."""
from typing import Any, Iterable, Mapping, Tuple, Union, Sequence
from typing_extensions import Protocol
import functools

from absl import logging
import chex
import jax
from jax import lax
import jax.numpy as jnp
import pyscf

from .ecp_potential import numerical_integral
from .ecp_potential import numerical_integral_optim

from .quadrature import get_quadrature
from .ph.hamiltonian import get_forward_laplacian_for_kinetic_ph

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]

class WavefunctionLike(Protocol):

  def __call__(self, params: ParamTree,
               electrons: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
    """

class LogWavefunctionLike(Protocol):

  def __call__(self, params: ParamTree, electrons: jnp.ndarray) -> jnp.ndarray:
    """Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
    """

class EnergyPattern(Protocol):

  def __call__(self, params: ParamTree, key: chex.PRNGKey,
               data: jnp.ndarray) -> jnp.ndarray:
    """Returns the local energy of a Hamiltonian at a configuration.
    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """


def get_dist(
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Constructs inputs to wavefunction from raw electron and atomic positions.

  Args:
    pos: electron positions. Shape (nelectrons*ndim,).
    atoms: atom positions. Shape (natoms, ndim).
    ndim: dimension of system. Change only with caution.

  Returns:
    r_ae, r_ee tuple, where:
      r_ae: atom-electron distance. Shape (nelectron, natom, 1).
      r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
    The diagonal terms in r_ee are masked out such that the gradients of these
    terms are also zero.
  """
  assert atoms.shape[1] == ndim
  ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
  ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])

  r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
  # Avoid computing the norm of zero, as is has undefined grad
  n = ee.shape[0]
  r_ee = (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))

  return r_ae, r_ee[..., None]

def local_kinetic_energy(
    f: LogWavefunctionLike,
    use_scan: bool = False,
    partition_num=0,
    forward_laplacian=True) -> LogWavefunctionLike:
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the log of the magnitude of the wavefunction.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    partition_num: 0: fori_loop implementation
                   1: Hessian implementation
                   other positive integer: Split the laplacian to multiple trunks and
                                           calculate accordingly.

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """
  vjvp = jax.vmap(jax.jvp, in_axes=(None, None, 0))

  def _forward_lapl_over_f(params, data):
      from lapjax import LapTuple, TupType
      output = f(params, LapTuple(data, is_input=True))
      return -0.5 * output.get(TupType.LAP) - \
              0.5 * jnp.sum(output.get(TupType.GRAD)**2)

  def _lapl_over_f(params, data):
    n = data.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.grad(f, argnums=1)
    grad_f_closure = lambda y: grad_f(params, y)
    # Use Hessian
    if partition_num == 1:
        g = grad_f_closure(data)
        f_closure = lambda x: f(params, x)
        hess = jax.hessian(f_closure)(data)
        return -0.5 * (jnp.trace(hess) + jnp.sum(g ** 2))

    # Original implementation
    if partition_num == 0:
        primal, dgrad_f = jax.linearize(grad_f_closure, data)

        if use_scan:
          _, diagonal = lax.scan(
              lambda i, _: (i + 1, dgrad_f(eye[i])[i]), 0, None, length=n)
          result = -0.5 * jnp.sum(diagonal)
        else:
          result = -0.5 * lax.fori_loop(
              0, n, lambda i, val: val + dgrad_f(eye[i])[i], 0.0)
        return result - 0.5 * jnp.sum(primal ** 2)

    # vjvp implementation
    assert n % partition_num == 0, f'partition_num {partition_num} does not divide the dimension {n}'
    eyes = jnp.asarray(jnp.array_split(eye, partition_num))
    def _body_fun(val, e):
        primal, tangent = vjvp(grad_f_closure, (data,), (e,))
        return val, (primal, tangent)
    _, (primal, tangent) = lax.scan(_body_fun, None, eyes)
    primal = primal.reshape((-1, primal.shape[-1]))
    tangent = tangent.reshape((-1, tangent.shape[-1]))
    return -0.5 * (jnp.sum(jnp.diagonal(primal) ** 2) + jnp.trace(tangent))

  if forward_laplacian:
    return _forward_lapl_over_f
  else:
    return _lapl_over_f


def potential_electron_electron(r_ee: jnp.ndarray) -> jnp.ndarray:
  """Returns the electron-electron potential.

  Args:
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
  """
  N = r_ee.shape[0]
  # This `jnp.eye` is a critical change if you want to take derivative of
  # local energy. Note that this doesn't introduce any bias at all.
  # It's a pure stability improvement for free.
  return jnp.sum(jnp.triu(1 / (r_ee[..., 0]+jnp.eye(N)), k=1))

def potential_electron_nuclear(charges: jnp.ndarray,
                               r_ae: jnp.ndarray) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
  """
  return -jnp.sum(charges / r_ae[..., 0])


def potential_nuclear_nuclear(charges: jnp.ndarray,
                              atoms: jnp.ndarray) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    atoms: Shape (natoms, ndim). Positions of the atoms.
  """
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  return jnp.sum(
      jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


def potential_energy(r_ae: jnp.ndarray, r_ee: jnp.ndarray, atoms: jnp.ndarray,
                     charges: jnp.ndarray) -> jnp.ndarray:
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
  return (potential_electron_electron(r_ee) +
          potential_electron_nuclear(charges, r_ae) +
          potential_nuclear_nuclear(charges, atoms))


def local_energy(f,
                 atoms: jnp.ndarray,
                 charges: jnp.ndarray,
                 use_scan: bool = False,
                 el_partition_num=0,
                 forward_laplacian=True,
                 ph_atoms=set(),
                 ph_info=None,
                 ph_rv_type='spline'):
  """Creates the function to evaluate the local energy.
  """
  log_abs_f = lambda *args, **kwargs: f(*args, **kwargs)[1]

  if len(ph_atoms) > 0:
    if not forward_laplacian:
        # (TODO) support non-Forward-Laplacian mode.
        raise NotImplementedError('Only supporting Forward Laplacian when considering PH')
    ph_atom_pos, raw_ph_data = ph_info
    ke = get_forward_laplacian_for_kinetic_ph(log_abs_f, raw_ph_data, ph_atom_pos,
                                              rv_type=ph_rv_type)
  else:
    ke = local_kinetic_energy(log_abs_f, use_scan=use_scan, partition_num=el_partition_num,
                              forward_laplacian=forward_laplacian)


  def _e_l(params, key, data) -> jnp.ndarray:
    del key  # unused

    r_ae, r_ee = get_dist(data, atoms)
    potential = potential_energy(r_ae, r_ee, atoms, charges)
    kinetic = ke(params, data)
    return potential + kinetic

  return _e_l

def ecp_potential(pe:jnp.ndarray, pa:jnp.ndarray, ecp_coe:dict) -> jnp.ndarray:
  """Returns the ecp potential. the form is equation 5 in the paper
    https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.013021

    Args:
      pe: position of electron
      pa: position of atoms
      ecp_dict: ecp parameter
    """
  norm = jnp.linalg.norm(pe[:,None,:] - pa,  axis=-1)
  res = []
  for _, l in ecp_coe:
    result = 0
    for power, coe in enumerate(l):
      for coeff in coe:
        result = result + norm[:, 0] ** (power - 2) * jnp.exp(-coeff[0] * norm[:, 0] ** 2) * \
                 coeff[1]
    res.append(result)
  res = jnp.stack(res, axis=-1)
  return res

def ecp_potential_select_core(pe:jnp.ndarray, pa:jnp.ndarray, ecp_coe:list, max_l:int) -> jnp.ndarray:
  rea_v = pe[:,None,:] - pa
  rea_norm = jnp.linalg.norm(rea_v,  axis=-1)
  res = []
  # Todo: optimize the for_loop in the future
  for l in range(max_l):
    result = 0
    if l < len(ecp_coe):
      for power, coe in enumerate(ecp_coe[l][1]):
        for coeff in coe:
          result = result + rea_norm ** (power - 2) * jnp.exp(-coeff[0] * rea_norm ** 2) * \
                 coeff[1]
    else:
      result = jnp.zeros_like(rea_norm)
    res.append(result)
  res = jnp.stack(res, axis=-1)
  return res, rea_v, rea_norm

def ecp_all(pe, element_coords, ecp, max_l):
    rea_v = []
    rea_norm = []
    ecp_res = []
    # Todo: optimize the for_loop in the future
    for sym, coord in element_coords.items():
      if sym in ecp:
        pa = jnp.array(coord)
        ecp_coe = ecp[sym][1]
        ecp_closure = functools.partial(ecp_potential_select_core,
                                        ecp_coe=ecp_coe,
                                        max_l=max_l)
        _ecp_res, _rea_v, _rea_norm = ecp_closure(pe, pa)
        rea_v.append(_rea_v)
        rea_norm.append(_rea_norm)
        ecp_res.append(_ecp_res)

    rea_v = jnp.concatenate(rea_v, axis=-2)
    rea_norm = jnp.concatenate(rea_norm, axis=-1)
    ecp_res = jnp.concatenate(ecp_res, axis=-2)
    index = jnp.argsort(rea_norm, axis=1)
    rea_v = jnp.take_along_axis(rea_v, index[..., None], axis=1)
    rea_norm = jnp.take_along_axis(rea_norm, index, axis=1)
    ecp_res = jnp.take_along_axis(ecp_res, index[..., None], axis=1)

    return ecp_res, rea_v, rea_norm

def non_local_energy(
                    fs: WavefunctionLike,
                    pyscf_mol,
                    ecp_element_list=None,
                    ecp_quadrature_id=None,
                    max_core=2,
                    ) -> EnergyPattern:
  quadrature = get_quadrature(ecp_quadrature_id)
  ecp_element_list = ecp_element_list or list(set(pyscf_mol.elements))
  element_coords = {}
  max_l = [len(s[1]) for s in list(pyscf_mol._ecp.values())]
  max_l = max(max_l)
  for ele in ecp_element_list:
        coord_list = []
        for sym, coord in pyscf_mol._atom:
            if sym == ele and sym in pyscf_mol._ecp:
                coord_list.append(jnp.asarray(coord)[None, ...])
        if len(coord_list) == 0:
          coord_list = None
          logging.warning('The element %s has not been assiged an ECP type', ele)
        else:
          coord_list = jnp.concatenate(coord_list, axis=0)
        element_coords[ele] = coord_list

  def non_local(pe, rea_v, rea, key, psi, func_state):
    res = numerical_integral(psi, rea_v, rea, pe, max_l, key, quadrature, func_state)
    return res / (4 * jnp.pi)

  def non_local_sum(params: ParamTree,
                    func_state,
                    key: chex.PRNGKey,
                    x:jnp.ndarray,
                    ) -> jnp.ndarray:
    """Returns the non-local energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    pe = x.reshape(-1,3)
    ecp_list, rea_v, rea = ecp_all(pe, element_coords, pyscf_mol._ecp, max_l)
    rea_v = rea_v[:, :max_core]
    rea = rea[:, :max_core]
    non_local_closure = functools.partial(non_local,
                                  psi=lambda x: fs(params, x.flatten()),)
    psi_int = non_local_closure(pe, rea_v, rea, key, func_state=func_state)
    integral_result = jnp.sum(ecp_list[..., :max_core, 1:] * psi_int
                                  , axis=-1)
    result = jnp.sum(integral_result) + jnp.sum(ecp_list[..., 0])
    return result

  return non_local_sum

def merge_ph_hph_info(ph_info, hph_info):
    """
    Merge PH and HPH information into a single (atom_pos, data) tuple.

    Rules:
    - If both are None, return None
    - If only PH exists, return PH
    - If only HPH exists, return HPH
    - If both exist, concatenate atom positions and merge data dicts
    """
    if ph_info is None and hph_info is None:
        return None

    if ph_info is None:
        return hph_info

    if hph_info is None:
        return ph_info

    ph_atom_pos_1, ph_data_1 = ph_info
    ph_atom_pos_2, ph_data_2 = hph_info

    merged_atom_pos = list(ph_atom_pos_1) + list(ph_atom_pos_2)
    merged_data = {**ph_data_1, **ph_data_2}

    return (merged_atom_pos, merged_data)

def pp_energy(f: WavefunctionLike,
               atoms: jnp.ndarray,
               nspins: Sequence[int],
               charges: jnp.ndarray,
               pyscf_mol: pyscf.gto.mole,
               pp_cfg,
               energy_local: EnergyPattern = None,
               use_scan: bool = False,
               el_partition_num=0,
               forward_laplacian=True,
               ) -> EnergyPattern:
  """Returns the total energy function.

  Args:
    f: network parameters.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
    pyscf_mol: pyscf.gto.mole.Mole class.
    pp_cfg: pseudopotential configuration. In particular:
       - ph_info: information for pure PH atoms (L2 only)
       - hph_info: information for HPH atoms (PH + ECP)
    Behavior:
       - atoms in ph_info contribute only through PH
       - atoms in hph_info contribute through both PH and ECP
       - atoms in pyscf_mol._ecp but not in either set contribute only through ECP
    energy_local: local energy function (parameter, key, position)
                if exist, use this local energy function,
                if None, generate the local energy in the program
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    el_partition_num: 0: fori_loop implementation
                      1: Hessian implementation
                      other positive integer: Split the laplacian to multiple chunks
    forward_laplacian: whether to use forward laplacian
  """
  ph_info = getattr(pp_cfg, 'ph_info', None)
  hph_info = getattr(pp_cfg, 'hph_info', None)

  ph_atoms = set() if ph_info is None else set(ph_info[1].keys())
  hph_atoms = set() if hph_info is None else set(hph_info[1].keys())
  all_ph_atoms = ph_atoms | hph_atoms

  logging.info(f'Elements for Pseudo-Hamiltonian: {ph_atoms}')
  logging.info(f'Elements for Hybrid Pseudo-Hamiltonian: {hph_atoms}')
  
  merged_ph_info = merge_ph_hph_info(ph_info, hph_info)
  
  ecp_atoms = []
  ecp_element_list = []
  
  # for logging only, in the case of pure ECP
  pure_ecp_atoms = set()

  for sym, coord in pyscf_mol._atom:
      if sym in pyscf_mol._ecp:
          if sym not in ph_atoms:
              ecp_atoms.append((sym, coord))
              ecp_element_list.append(sym)
          if sym not in all_ph_atoms:
              pure_ecp_atoms.add(sym)

  logging.info(f'Elements for ECP: {pure_ecp_atoms}')
  
  ecp_quadrature_id = pp_cfg.ecp_quadrature_id
  max_core = pp_cfg.ecp_select_core.max_core

  if energy_local is None or len(all_ph_atoms) > 0:
    energy_local = local_energy(
        f,
        atoms,
        charges,
        use_scan,
        el_partition_num,
        forward_laplacian=forward_laplacian,
        ph_atoms=all_ph_atoms,
        ph_info=merged_ph_info,
        ph_rv_type=pp_cfg.ph_rv_type,
    )
    logging.info('Using local energy from JaQMC implementation')

  if len(ecp_element_list) > 0:
      energy_nonlocal = non_local_energy(
          f,
          pyscf_mol,
          ecp_element_list,
          ecp_quadrature_id,
          max_core,
      )
      if hph_atoms:
          logging.info('Hybrid Pseudo-Hamiltonian atoms detected. Adding non-local terms')
      else:
          logging.info('ECP atoms detected. Adding non-local terms')
  else:
      energy_nonlocal = lambda *args, **kwargs: 0.0
      logging.info('No non-local term added')

  rearrange = [jnp.array([0]),                  # spin-up start
               jnp.array([nspins[0]]),          # spin-up end
               jnp.array([nspins[0]]),          # spin-down start
               jnp.array([nspins[0] + nspins[1]])]  # spin-down end

  energy_ecp = lambda params, key, x: energy_local(params, key, x) \
                + energy_nonlocal(params, rearrange, key, x)
  return energy_ecp

def non_local_energy_optim(
                    f_modify,
                    f_memory,
                    f_memory_update,
                    pyscf_mol,
                    ecp_element_list=None,
                    ecp_quadrature_id=None,
                    max_core=1
                    ) -> EnergyPattern:
  quadrature = get_quadrature(ecp_quadrature_id)
  ecp_element_list = ecp_element_list or list(set(pyscf_mol.elements))
  logging.info(f'ecp elements: {ecp_element_list}')
  element_coords = {}
  max_l = [len(s[1]) for s in list(pyscf_mol._ecp.values())]
  max_l = max(max_l)
  for ele in ecp_element_list:
        coord_list = []
        for sym, coord in pyscf_mol._atom:
            if sym == ele and sym in pyscf_mol._ecp:
                coord_list.append(jnp.asarray(coord)[None, ...])
        if len(coord_list) == 0:
          coord_list = None
        else:
          coord_list = jnp.concatenate(coord_list, axis=0)
        element_coords[ele] = coord_list

  def non_local(pe, rea_v, rea, key, psi_modify, psi_memory, psi_memory_update, func_state):
    res = numerical_integral_optim(psi_modify, psi_memory, psi_memory_update, rea_v, rea, pe, max_l, key, quadrature, func_state)
    return res / (4 * jnp.pi)

  def non_local_sum(params: ParamTree,
                    func_state,
                    key: chex.PRNGKey,
                    x:jnp.ndarray) -> jnp.ndarray:
    """Returns the non-local energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    pe = x.reshape(-1,3)
    ecp_list, rea_v, rea = ecp_all(pe, element_coords, pyscf_mol._ecp, max_l)
    rea_v = rea_v[:, :max_core]
    rea = rea[:, :max_core]
    non_local_closure = functools.partial(non_local,
                          psi_modify=lambda memory, x : f_modify(params, memory, x.flatten()),
                          psi_memory=lambda x : f_memory(params, x.flatten()),
                          psi_memory_update=lambda memory, x, e_index : f_memory_update(params, memory, x.flatten(), e_index)
                          )
    psi_int = non_local_closure(pe, rea_v, rea, key, func_state=func_state)
    integral_result = jnp.sum(ecp_list[..., :max_core, 1:] * psi_int
                                  , axis=-1)
    result = jnp.sum(integral_result) + jnp.sum(ecp_list[..., 0])
    return result

  return non_local_sum


def ecp_energy_optim(
                     f: WavefunctionLike,
                     f_modify,
                     f_memory,
                     f_memory_update,
                     atoms: jnp.ndarray,
                     nspins: Sequence[int],
                     charges: jnp.ndarray,
                     pyscf_mol: pyscf.gto.mole,
                     pp_cfg,
                     energy_local : EnergyPattern = None,
                     use_scan: bool = False,
                     el_partition_num = 0,
                     forward_laplacian=True,
                    ) -> EnergyPattern:
  """Returns the total energy funtion.

  Args:
    f: network parameters.
    f_modify: wavefunction network value depend on network memory.
    f_memory: The part of network only depend on one electron position.
    f_memory_update: update value of memory by one electron move.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    pyscf_mol: pyscf.gto.mole.Mole class.
    pp_cfg: pp config.
    charges:Shape (natoms). Nuclear charges of the atoms.
    ecp_quadrature_id: Expected to be "quadrature_type" + "_" + "number of points
            used in quadrature". It could also be None, in which case a default one
            will be used.
    energy_local: local energy function (parameter, key, position)
                if exist, use this local energy function,
                if None,  generate the local energy in the program
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    el_partition_num: 0: fori_loop implementation
                      1: Hessian implementation
                      other positive integer: Split the laplacian to multiple trunks and
                                           calculate accordingly.
  """
  if pp_cfg.ph_info is None:
      ph_atoms = set()
  else:
      ph_atoms = set(pp_cfg.ph_info[1].keys())
  logging.info(f'Elements for Pseudo-Hamiltonian: {ph_atoms}')

  ecp_atoms = []
  ecp_element_list = []
  for sym, coord in pyscf_mol._atom:
      if sym in pyscf_mol._ecp and sym not in ph_atoms:
          ecp_atoms.append((sym, coord))
          ecp_element_list.append(sym)
  logging.info(f'Elements for ECP: {set(x[0] for x in ecp_atoms)}')

  ecp_quadrature_id = pp_cfg.ecp_quadrature_id
  max_core = pp_cfg.ecp_select_core.max_core

  if energy_local is None or len(ph_atoms) > 0:
    energy_local = local_energy(
        f,
        atoms,
        charges,
        use_scan,
        el_partition_num,
        forward_laplacian=forward_laplacian,
        ph_atoms=ph_atoms,
        ph_info=pp_cfg.ph_info,
        ph_rv_type=pp_cfg.ph_rv_type,
    )
    logging.info(f'Using local energy from JaQMC implementation')

  if len(ecp_element_list) > 0:
      energy_nonlocal = non_local_energy_optim(
          f_modify,
          f_memory,
          f_memory_update,
          pyscf_mol,
          ecp_element_list,
          ecp_quadrature_id,
          max_core,
      )
      logging.info(f'ECP atoms detected. Adding non-local terms')
  else:
      energy_nonlocal = lambda *args, **kwargs: 0.0
      logging.info(f'No ECP atoms detected. No non-local term added')

  #TODO: Currently, rearranging is not yet functional; we will add the corresponding feature in the future.
  rearrange = [jnp.array([0]),# spin-up start
               jnp.array([nspins[0]]),# spin-up end
               jnp.array([nspins[0]]),# spin-down start
               jnp.array([nspins[0] + nspins[1]]),# spin-down end
               ]
  energy_ecp = lambda params, key, x: energy_local(params, key, x) \
                                  + energy_nonlocal(params, rearrange, key, x)
  return energy_ecp