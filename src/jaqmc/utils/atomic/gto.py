# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/google-deepmind/ferminet/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""Evaluate Gaussian-Type Orbitals on a grid using JAX."""

import collections
from collections.abc import Mapping
from typing import Any, Self

import jax
import jax.scipy.special as jss
import numpy as np
import pyscf.gto
import pyscf.pbc.gto
from jax import numpy as jnp
from pyscf.lib import exceptions


def normalize_primitive_weights(basis_list):
    """Correctly handle the primitive weighting and normalization.

    A general basis_list specification for a cGTO of the form
        [L, [alpha_1, w_1], [alpha_2, w_2], ...]
    may not correspond to a basis function that square integrates to 1 unless the
    w_i's are correctly scaled. pyscf.gto.mole.make_bas_env performs the
    necessary scaling of the w_i's, so we execute make_bas_env and read back the
    correctly scaled basis_list from the output.

    Args:
        basis_list: a list of cGTO basis specifications of the form
            [L, [alpha_1, w_1], [alpha_2, w_2], ...]

    Returns:
        a basis_list where all w_i's are correctly scaled to ensure that the cGTOs
        square integrate to 1.
    """
    bas, env = pyscf.gto.mole.make_bas_env(basis_list)
    bas = np.array(bas)
    angl = bas[:, 1]
    start_ptrs = bas[:, 5]
    spec_shape = bas[:, 3:1:-1] + [[1, 0]]
    stop_ptrs = start_ptrs + spec_shape[:, 0] * spec_shape[:, 1]
    basis_list = []
    for ell, start, stop, shape in zip(angl, start_ptrs, stop_ptrs, spec_shape):
        basis_list.append([ell, *env[start:stop].reshape(shape).T.tolist()])
    return basis_list


def get_basis_for_atom(mol_basis, atom):
    if isinstance(mol_basis, str):
        atom_basis = pyscf.gto.basis.load(mol_basis, atom)
    elif isinstance(mol_basis, dict) and isinstance(mol_basis[atom], str):
        try:
            atom_basis = pyscf.gto.basis.load(mol_basis[atom], atom)
        except exceptions.BasisNotFoundError:
            atom_basis = mol_basis[atom]
    else:
        atom_basis = mol_basis[atom]
    return normalize_primitive_weights(atom_basis)


def cart2sph(r: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert Cartesian coordinates to spherical coordinates.

    Args:
        r: Cartesian coordinates with ``x``, ``y``, and ``z`` on the last axis.

    Returns:
        Tuple ``(rho, phi, theta)`` where ``rho`` is the radial distance,
        ``phi`` is the azimuthal angle in the ``xy`` plane, and ``theta`` is
        the polar angle measured from the positive ``z`` axis.
    """
    rho = jnp.linalg.norm(r, axis=-1)
    phi = jnp.arctan2(r[..., 1], r[..., 0])
    theta = jnp.arctan2(jnp.linalg.norm(r[..., :2], axis=-1), r[..., 2])
    return rho, phi, theta


def solid_harmonic(r: jnp.ndarray, l_max: int) -> jnp.ndarray:
    """Computes all solid harmonics r**ell Y_{ell, m} for all ell <= l_max.

    Returns:
        jnp.ndarray: all solid harmonics r**ell Y_{ell, m} for all ell <= l_max.
    """
    r_scalar, phi, theta = cart2sph(r)
    cos_theta = jnp.cos(theta)
    legendre = jss.lpmn_values(l_max, l_max, cos_theta, True)

    m = jnp.arange(l_max + 1)[:, None, None]
    angle = m * phi[None, None, :]
    sign = (-1) ** m[1:]
    positive_harmonics = np.sqrt(2) * (legendre[1:] * jnp.cos(angle[1:])) * sign
    zero_harmonics = legendre[0:1]
    negative_harmonics = np.sqrt(2) * ((legendre[1:] * jnp.sin(angle[1:])) * sign)[::-1]
    harmonics = jnp.concatenate(
        [negative_harmonics, zero_harmonics, positive_harmonics], axis=0
    )

    ell = jnp.arange(l_max + 1)[None, :, None]
    return harmonics * (r_scalar[None, None, :] ** ell)


class AtomicOrbitalEvaluator:
    """An atomic orbital evaluator using GTO basis in JAX.

    Attributes:
        atom_list: A list of tuples (Z, [x, y, z]) that specifies the atom types (Z)
            and positions (x,y,z) in Bohr for the Mol
        basis_dict: A dictionary of cGTO specifications of the form
            {"Z": [[L, [alpha_1, w_1], [alpha_2, w_2], ...], ...], ...}
        spec: The dict returned by _get_orbital_construction_dict().
    """

    def __init__(self, atom_list, basis_dict):
        self.atom_list = atom_list
        self.basis_dict = basis_dict
        self._spec = self._get_orbital_construction_dict()
        # Needed to provide a concrete value to segment_sum in eval_gto,
        # otherwise tracing + pmap will not work properly
        self._num_segments = self._spec["cshell_id"].max() + 1

    @property
    def spec(self) -> Mapping[str, Any]:
        return self._spec

    @classmethod
    def from_pyscf(cls, mol: pyscf.gto.Mole) -> Self:
        """Initialize from a pyscf.gto.Mole.

        Returns:
            The initialized instance.
        """
        atom_list = pyscf.gto.format_atom(mol.atom, unit=mol.unit)
        basis_dict = {
            atom: get_basis_for_atom(mol.basis, atom)
            for atom in set(next(zip(*atom_list)))
        }
        return cls(atom_list, basis_dict)

    def _get_max_l(self):
        return np.max(
            [
                basis[0]  # pylint: disable=g-complex-comprehension
                for atom in self.atom_list
                for basis in self.basis_dict[atom[0]]
            ]
        )

    def _get_orbital_construction_dict(self):
        """Creates description of all primitive orbitals to be constructed.

        Specifically, this function unravels a contracted basis set description into
        a number of lists that organize construction of pGTOs and contraction to
        cGTOs

        We define 3 concepts:
            pGTO: a primitive GTO of rhe form N exp(-alpha r^2) r^l Y_lm(r)
            cGTO: a contracted GTO which consists of a sum of pGTOs with the same l
                and m but different gaussian precisions a and weightings N
            pShell: a set of all pGTOs consisting of all possible m for orbitals with
                a fixed radial part and angular momentum l
            cShell: like a pShell but made from cGTOs with all m.

        Note that every pGTO requires specification of a radial part and an angular
        part. The radial part is identical for all pGTOs in a pShell.

        The lists are as follows:
            alpha[i]: the precision of the gaussian part of the i^th pShell
            primitive_weights[i]: the weighting N of the i^th pShell including any
                factor needed to normalize the cGTOs to square integrate to 1 after
                contraction.
            atom_id[i]: the index of the atom on which the i^th pShell sits
            cshell_id[i]: the index of the cShell into which the i^th pShell will be
                contracted
            l[i]: the orbital angular momentum of the i^th pShell
            radial_index[i]: the index of the cShell to use for the radial part of
                the i^th cGTO
            angular_index[i]: a tuple containing of the orbital angular momentum, the
                azimuthal quantum number and the atom_id of the i^th cGTO

        Additionally the dictionary contains
            atom_centres[i]: the x,y,z, coordinate of the i^th atom
            l_max: the largest orbital angular momentum in the basis

        Returns:
            A dictionary of the construction lists described above
        """
        construction_spec = collections.defaultdict(list)
        cshell_id = 0
        for atom_id, atom in enumerate(self.atom_list):
            for basis in self.basis_dict[atom[0]]:
                ell = basis[0]
                n_weights = len(basis[1]) - 1
                for w in range(n_weights):
                    for info in basis[1:]:
                        # Only store an uncontracted exponent if it is actually used
                        # in a basis set.
                        if abs(info[w + 1]) > 1e-12:
                            construction_spec["alpha"].append(info[0])
                            construction_spec["primitive_weights"].append(info[w + 1])
                            construction_spec["atom_id"].append(atom_id)
                            construction_spec["cshell_id"].append(cshell_id)
                            construction_spec["l"].append(ell)
                    construction_spec["radial_index"] += [
                        cshell_id for _ in range(2 * ell + 1)
                    ]
                    ms = (
                        [1, -1, 0] if ell == 1 else range(-ell, ell + 1)
                    )  # pyscf reorders p orbitals
                    for m in ms:
                        construction_spec["angular_index"].append((ell, m, atom_id))
                    cshell_id += 1
        construction_spec["atom_centres"] = np.array(list(zip(*self.atom_list))[1])
        for k, v in construction_spec.items():
            construction_spec[k] = np.array(v)
        return construction_spec

    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        r"""Computes all gtos on the grid of coords.

        A primitive GTO consists of the product of a gaussian radial part and a
        solid harmonic angular part

            N exp(-alpha r^2) r^l Y_lm(r)
            \--------------/  \--------/
                radial part     angular part

        The radial part is the same for all elements in a cShell, so we first
        construct all radial part for all cShells and then join these with the
        angular parts as dictated by the lists in self._spec.

        Args:
            coords: a [G, 3] array containing the xyz coords at which to evaluate the
                GTOs

        Returns:
            A [G, CGTO] array of the evaluated GTOs.

        Notes:
            Shape annotations:
                G = len(coords),
                A = number of atoms
                L = max_l
                PGTO = number of primitive GTOs
                CGTO = mol.nao_nr() (number of contrated GTOs)
                CSHELL = mol.nbas = number of contracted shells
        """
        # construct copies of the grid centred on each atom [G, A, 3]
        dr = coords[:, None, :] - self._spec["atom_centres"]
        flat_dr = dr.reshape(-1, 3)
        # construct all solid harmonics [2L+1, L, (G*A)]
        max_l = self._get_max_l()
        sh = solid_harmonic(flat_dr, max_l)
        sh = sh.reshape(sh.shape[0], sh.shape[1], dr.shape[0], dr.shape[1])

        # construct the radial part
        r_sqr = jnp.linalg.norm(dr, axis=-1) ** 2  # [G, A]
        g = (
            jnp.exp(-self._spec["alpha"][None, :] * r_sqr[:, self._spec["atom_id"]])
            * self._spec["primitive_weights"][None, :]
        )  # [G, PGTO]

        # contract the primitives
        g = jax.ops.segment_sum(  # [CSHELL, G]
            g.T, self._spec["cshell_id"], num_segments=self._num_segments
        )
        radial_part = g[self._spec["radial_index"]]  # [CGTO, G]

        angular_part = sh[
            self._spec["angular_index"][:, 1] + max_l,
            self._spec["angular_index"][:, 0],
            :,
            self._spec["angular_index"][:, 2],
        ]

        return (angular_part * radial_part).T  # [G, CGTO]


class PBCAtomicOrbitalEvaluator:
    r"""Evaluator for Bloch-summed Gaussian-type orbitals in periodic systems.

    This class constructs crystal orbitals suitable for periodic boundary
    conditions by summing localized atomic orbitals over lattice translation
    vectors with k-point-dependent phase factors. The resulting orbitals
    satisfy Bloch's theorem and can be used for k-point sampling in solid-state
    calculations.

    The evaluator wraps an :class:`AtomicOrbitalEvaluator` for the underlying
    GTO evaluation and handles the lattice summation internally.

    Attributes:
        eval_aos: The underlying :class:`AtomicOrbitalEvaluator` for computing
            localized atomic orbitals.
        lattice_vectors: Array of shape ``[L, 3]`` containing lattice translation
            vectors :math:`\mathbf{L}` used in the Bloch sum, sorted by distance
            from the origin.

    Example::

        cell = pyscf.pbc.gto.Cell(atom="H 0 0 0", a=np.eye(3) * 4, basis="sto-3g")
        cell.build()
        kpts = cell.make_kpts([2, 2, 2])
        evaluator = PBCAtomicOrbitalEvaluator.from_pyscf(cell)
        aos = evaluator(coords, kpts=kpts)  # shape: [nk, ncoords, nao]
    """

    def __init__(self, atom_list, basis_dict, image_translation_vectors: jnp.ndarray):
        self.eval_aos = AtomicOrbitalEvaluator(atom_list, basis_dict)
        self.image_translation_vectors = image_translation_vectors

    @classmethod
    def from_pyscf(cls, cell: pyscf.pbc.gto.Cell, rcut: float | None = None) -> Self:
        atom_list = pyscf.gto.format_atom(cell.atom, unit=cell.unit)
        basis_dict = {
            atom: get_basis_for_atom(cell.basis, atom)
            for atom in set(next(zip(*atom_list)))
        }
        # Get lattice translation vectors within cutoff
        rcut = rcut or float(pyscf.pbc.gto.estimate_rcut(cell))
        image_translation_vectors = cell.get_lattice_Ls(rcut=rcut)
        # Sort by distance for better numerical convergence
        image_distances = jnp.linalg.norm(image_translation_vectors, axis=-1)
        sorted_indices = jnp.argsort(image_distances)
        return cls(atom_list, basis_dict, image_translation_vectors[sorted_indices])

    def __call__(self, coords: jnp.ndarray, kpts: jnp.ndarray) -> jnp.ndarray:
        r"""Evaluate Bloch-summed atomic orbitals at given coordinates and k-points.

        Constructs crystal orbitals by summing atomic orbitals over all periodic
        images with appropriate phase factors:

        .. math::

            \phi_{\mathbf{k}}(\mathbf{r}) = \sum_{\mathbf{L}}
                e^{i\mathbf{k} \cdot \mathbf{L}} \chi(\mathbf{r} - \mathbf{L})

        where :math:`\mathbf{L}` are lattice translation vectors and
        :math:`\chi(\mathbf{r} - \mathbf{L})` is the atomic orbital centered at
        :math:`\mathbf{L}`. The resulting Bloch orbital satisfies
        :math:`\phi_{\mathbf{k}}(\mathbf{r} + \mathbf{R}) =
        e^{i\mathbf{k} \cdot \mathbf{R}} \phi_{\mathbf{k}}(\mathbf{r})`.

        Args:
            coords: Electron coordinates with shape ``[ncoords, 3]``.
            kpts: K-points with shape ``[nk, 3]``.

        Returns:
            Complex array of shape ``[nk, ncoords, nao]`` containing the Bloch-summed
            atomic orbitals evaluated at each k-point.
        """
        # Evaluate AOs at (r - L) for each lattice vector L
        image_coords = coords[None, :, :] - self.image_translation_vectors[:, None, :]
        image_gtos = jax.vmap(self.eval_aos)(image_coords)
        image_phases = jnp.exp(1j * kpts @ self.image_translation_vectors.T)
        return jnp.einsum("kl,lna->kna", image_phases, image_gtos)
