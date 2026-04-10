# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/google-deepmind/ferminet/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""Interaction with Hartree-Fock solver in pyscf."""

# Abbreviations used:
# SCF: self-consistent field (method). Another name for Hartree-Fock
# HF: Hartree-Fock method.
# RHF: restricted Hartre-Fock. Require molecular orbital for the i-th alpha-spin
#   and i-th beta-spin electrons to have the same spatial component.
# ROHF: restricted open-shell Hartree-Fock. Same as RHF except allows the number
#   of alpha and beta electrons to differ.
# UHF: unrestricted Hartre-Fock. Permits breaking of spin symmetry and hence
#   alpha and beta electrons to have different spatial components.
# AO: Atomic orbital. Underlying basis set (typically Gaussian-type orbitals and
#   built into pyscf).
# MO: molecular orbitals/Hartree-Fock orbitals. Single-particle orbitals which
#   are solutions to the Hartree-Fock equations.

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pyscf.gto
import pyscf.lib
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.scf
from jax import numpy as jnp

from jaqmc.geometry.pbc import wrap_positions

from .atom import Atom
from .gto import AtomicOrbitalEvaluator, PBCAtomicOrbitalEvaluator

NDArray = jnp.ndarray | np.ndarray
logger = logging.getLogger(__name__)


def _extract_spin_blocks(
    alpha_mos: NDArray,
    beta_mos: NDArray,
    nspins: tuple[int, int],
    leading_dims: tuple[int, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract spin-block orbital matrices from MO arrays.

    Reshapes flattened MO arrays and extracts the occupied orbital blocks
    for alpha and beta spins using the Aufbau principle.

    Args:
        alpha_mos: Alpha MO values, shape (batch*nelec, norbs).
        beta_mos: Beta MO values, shape (batch*nelec, norbs).
        nspins: Tuple of (n_alpha, n_beta) electrons.
        leading_dims: Original leading dimensions of the position array.

    Returns:
        Tuple of (alpha_matrix, beta_matrix) with shapes
        (*leading_dims, n_alpha, n_alpha) and (*leading_dims, n_beta, n_beta).
    """
    n_alpha, n_beta = nspins
    nelec = n_alpha + n_beta
    alpha_mos = jnp.reshape(alpha_mos, (*leading_dims, nelec, -1))
    beta_mos = jnp.reshape(beta_mos, (*leading_dims, nelec, -1))
    alpha_matrix = alpha_mos[..., :n_alpha, :n_alpha]
    beta_matrix = beta_mos[..., n_alpha:, :n_beta]
    return alpha_matrix, beta_matrix


def _eval_slater_from_orbitals(
    alpha_matrix: NDArray, beta_matrix: NDArray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Slater determinant sign and log|det| from orbital matrices.

    Args:
        alpha_matrix: Alpha orbital matrix, shape (..., n_alpha, n_alpha).
        beta_matrix: Beta orbital matrix, shape (..., n_beta, n_beta).

    Returns:
        Tuple of (sign, log_abs_det) for the Slater determinant.
    """
    sign_alpha, logdet_alpha = jnp.linalg.slogdet(alpha_matrix)
    sign_beta, logdet_beta = jnp.linalg.slogdet(beta_matrix)
    return sign_alpha * sign_beta, logdet_alpha + logdet_beta


class MolecularSCF:
    """Helper class for running Hartree-Fock (self-consistent field) with pyscf.

    Attributes:
        molecule: list of system.Atom objects giving the atoms in the
            molecule and their positions.
        nelectrons: Tuple with number of alpha electrons and beta
            electrons.
        basis: Basis set to use, best specified with the relevant string
            for a built-in basis set in pyscf. A user-defined basis set can be used
            (advanced). See https://sunqm.github.io/pyscf/gto.html#input-basis for
            more details.
        pyscf_mol: the PySCF 'Molecule'. If this is passed to the init,
            the molecule, nelectrons, and basis will not be used, and the
            calculations will be performed on the existing pyscf_mol
        restricted: If true, use the restricted Hartree-Fock method, otherwise use
            the unrestricted Hartree-Fock method.
        mean_field: the actual UHF object.
        mo_coeff: The molecular orbital coefficients computed by Hartree-Fock.
    """

    def __init__(
        self,
        molecule: Sequence[Atom] | None = None,
        nelectrons: tuple[int, int] | None = None,
        basis: str | Mapping[str, str] | None = "cc-pVTZ",
        ecp: str | Mapping[str, str] | None = None,
        core_electrons: Mapping[str, int] | None = None,
        pyscf_mol: pyscf.gto.Mole | None = None,
        restricted: bool = True,
    ):
        pyscf.lib.param.TMPDIR = None

        if pyscf_mol:
            self._mol = pyscf_mol
        else:
            # If not passed a pyscf molecule, create one
            assert molecule is not None
            assert nelectrons is not None
            if any(atom.atomic_number - atom.charge > 1.0e-8 for atom in molecule):
                logger.info(
                    "Fractional nuclear charge detected. "
                    "Running SCF on atoms with integer charge."
                )
            ecp = ecp or {}
            core_electrons = core_electrons or {}

            nuclear_charge = 0
            for atom in molecule:
                nuclear_charge += atom.atomic_number
                if atom.symbol in core_electrons:
                    nuclear_charge -= core_electrons[atom.symbol]
            charge = nuclear_charge - sum(nelectrons)
            self._mol = pyscf.gto.Mole(
                atom=[[atom.symbol, atom.coords] for atom in molecule], unit="bohr"
            )
            self._mol.basis = basis
            self._mol.spin = nelectrons[0] - nelectrons[1]
            self._mol.charge = charge
            self._mol.ecp = ecp
            self._mol.build()
            if self._mol.nelectron != sum(nelectrons):
                raise RuntimeError("PySCF molecule not consistent with QMC molecule.")
            self.eval_aos = AtomicOrbitalEvaluator.from_pyscf(self._mol)
        if restricted:
            self.mean_field = pyscf.scf.RHF(self._mol)
        else:
            self.mean_field = pyscf.scf.UHF(self._mol)

        # Create pure-JAX Mol object so that GTOs can be evaluated in traced
        # JAX functions
        self.eval_aos = AtomicOrbitalEvaluator.from_pyscf(self._mol)
        self.restricted = restricted

    def run(self, dm0: np.ndarray | None = None):
        """Runs the Hartree-Fock calculation.

        Args:
            dm0: Optional density matrix to initialize the calculation.

        Returns:
            A pyscf scf object (i.e. pyscf.scf.rhf.RHF, pyscf.scf.uhf.UHF or
            pyscf.scf.rohf.ROHF depending on the spin and restricted settings).
        """
        try:
            self.mean_field.kernel(dm0=dm0)
        except TypeError:
            logger.info(
                "Mean-field solver does not support specifying an initial "
                "density matrix."
            )
            # 1e solvers (e.g. uhf.HF1e) do not take any keyword arguments.
            self.mean_field.kernel()
        return self.mean_field

    def eval_mos(self, positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluates the Hartree-Fock single-particle orbitals at a set of points.

        Args:
            positions: numpy array of shape (N, 3) of the positions in space at which
                to evaluate the Hartree-Fock orbitals.

        Returns:
            Pair of numpy float64 arrays of shape (N, M) (deriv=False) or (4, N, M)
            (deriv=True), where 2M is the number of Hartree-Fock orbitals. The (i-th,
            j-th) element in the first (second) array gives the value of the j-th
            alpha (beta) Hartree-Fock orbital at the i-th electron position in
            positions. For restricted (RHF, ROHF) calculations, the two arrays will be
            identical.
            If deriv=True, the first index contains [value, x derivative, y
            derivative, z derivative].

        Raises:
            RuntimeError: If Hartree-Fock calculation has not been performed using
                `run`.
            NotImplementedError: If Hartree-Fock calculation used Cartesian
                Gaussian-type orbitals as the underlying basis set.
        """
        if self.mean_field is None:
            raise RuntimeError("Mean-field calculation has not been run.")
        if self.restricted:
            coeffs = (self.mean_field.mo_coeff,)
        else:
            coeffs = self.mean_field.mo_coeff
        # Assumes self._mol.cart (use of Cartesian Gaussian-type orbitals and
        # integrals) is False (default behaviour of pyscf).
        if self._mol.cart:
            raise NotImplementedError(
                "Evaluation of molecular orbitals using cartesian GTOs."
            )
        ao_values = self.eval_aos(jnp.asarray(positions))
        mo_values = tuple(jnp.matmul(ao_values, coeff) for coeff in coeffs)
        if self.restricted:
            # duplicate for beta electrons.
            mo_values *= 2
        return mo_values  # type: ignore[return-value]

    def eval_orbitals(
        self, pos: jnp.ndarray, nspins: tuple[int, int]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluates SCF orbitals at a set of positions.

        Args:
            pos: an array of electron positions to evaluate the orbitals at, of shape
                (..., nelec, 3), where the leading dimensions are arbitrary, nelec is
                the number of electrons and the spin up electrons are ordered before
                the spin down electrons.
            nspins: tuple with number of spin up and spin down electrons.

        Returns:
            tuple with matrices of orbitals for spin up and spin down electrons, with
            the same leading dimensions as in pos.

        Raises:
            ValueError: If input is not a NumPy or JAX array.
        """
        if not isinstance(pos, np.ndarray):  # works even with JAX array
            try:
                pos = pos.copy()
            except AttributeError as exc:
                raise ValueError("Input must be either NumPy or JAX array.") from exc
        leading_dims = pos.shape[:-2]
        # split into separate electrons
        pos = jnp.reshape(pos, [-1, 3])  # (batch*nelec, 3)
        alpha_mos, beta_mos = self.eval_mos(pos)  # (batch*nelec, nbasis) each
        return _extract_spin_blocks(alpha_mos, beta_mos, nspins, leading_dims)

    def eval_slater(
        self, pos: jnp.ndarray, nspins: tuple[int, int]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate Slater determinant sign and log|det|.

        Args:
            pos: An array of electron positions to evaluate the orbitals at.
            nspins: Tuple with number of spin up and spin down electrons.

        Returns:
            Tuple with sign and log absolute value of Slater determinant.
        """
        alpha_matrix, beta_matrix = self.eval_orbitals(pos, nspins)
        return _eval_slater_from_orbitals(alpha_matrix, beta_matrix)


class PeriodicSCF:
    r"""Helper class for running periodic Hartree-Fock with k-point sampling.

    This class handles the additional complexity of periodic systems compared
    to :class:`MolecularSCF`:

    1. **K-point sampling**: Electrons occupy Bloch states at different k-points
       in the Brillouin zone. The ``kpts`` parameter specifies the k-point mesh.

    2. **Bloch phase corrections**: When electron coordinates are wrapped into
       the primitive cell, a phase factor :math:`e^{i\mathbf{k}\cdot\mathbf{R}}`
       must be applied (see :meth:`eval_mos` for details).

    3. **Variable occupancy**: Different k-points may have different numbers of
       occupied orbitals due to band filling. This requires pre-extracting MO
       coefficients via ``_extract_mo_coeffs`` and padding for array ops.

    4. **K-point assignment**: The ``klist`` parameter in :meth:`eval_orbitals`
       specifies which k-point's orbitals to use for each electron, enabling
       twist-averaged boundary conditions.

    Args:
        atoms: List of Atom objects giving atom types and positions.
        nelectrons: Tuple of ``(n_alpha, n_beta)`` electrons per primitive cell.
        basis: Basis set string (e.g., ``"cc-pVTZ"``).
        lattice_vectors: Primitive cell lattice vectors, shape ``(3, 3)``.
        kpts: K-points for sampling, shape ``(nk, 3)``. If ``None``, uses
            Gamma point only.
        ecp: Effective core potentials mapping.
        core_electrons: Core electrons mapping.
        pyscf_cell: Pre-built PySCF Cell. If provided, atoms/basis/lattice are
            ignored.
        restricted: Use restricted (``True``) or unrestricted (``False``) HF.
        rcut: Cutoff radius for lattice sum. If ``None``, uses PySCF estimate.
    """

    def __init__(
        self,
        atoms: Sequence[Atom] | None = None,
        nelectrons: tuple[int, int] | None = None,
        basis: str | Mapping[str, str] | None = "cc-pVTZ",
        lattice_vectors: NDArray | None = None,
        kpts: NDArray | None = None,
        ecp: str | Mapping[str, str] | None = None,
        core_electrons: Mapping[str, int] | None = None,
        pyscf_cell: pyscf.pbc.gto.Cell | None = None,
        restricted: bool = True,
        rcut: float | None = None,
    ):
        pyscf.lib.param.TMPDIR = None

        if pyscf_cell is not None:
            self._cell = pyscf_cell
        else:
            assert atoms is not None
            assert nelectrons is not None
            assert lattice_vectors is not None

            ecp = ecp or {}
            core_electrons = core_electrons or {}

            # Calculate charge (same logic as MolecularSCF)
            nuclear_charge = 0
            for atom in atoms:
                nuclear_charge += atom.atomic_number
                if atom.symbol in core_electrons:
                    nuclear_charge -= core_electrons[atom.symbol]
            charge = nuclear_charge - sum(nelectrons)

            self._cell = pyscf.pbc.gto.Cell(
                atom=[[atom.symbol, atom.coords] for atom in atoms],
                a=lattice_vectors,
                unit="bohr",
            )
            self._cell.basis = basis
            self._cell.spin = nelectrons[0] - nelectrons[1]
            self._cell.charge = charge
            self._cell.ecp = ecp
            self._cell.build()

        # Set up k-points
        self.kpts = jnp.asarray(kpts) if kpts is not None else jnp.zeros((1, 3))

        # Create mean field object with density fitting for efficiency.
        if restricted:
            self.mean_field = pyscf.pbc.scf.KRHF(
                self._cell, exxdiv="ewald", kpts=np.asarray(self.kpts)
            ).density_fit()
        else:
            self.mean_field = pyscf.pbc.scf.KUHF(
                self._cell, exxdiv="ewald", kpts=np.asarray(self.kpts)
            ).density_fit()

        self.restricted = restricted
        self.eval_aos = PBCAtomicOrbitalEvaluator.from_pyscf(self._cell, rcut=rcut)
        self._mo_coeff: tuple[list, list] | None = None

    def run(self, dm0: np.ndarray | None = None):
        """Run the k-point HF calculation.

        Args:
            dm0: Optional density matrix to initialize the calculation.

        Returns:
            The PySCF mean field object after SCF convergence.
        """
        logger.info("Start %s", type(self.mean_field).__name__)
        self.mean_field.kernel(dm0=dm0)
        self._extract_mo_coeffs()
        logger.info("Complete %s", type(self.mean_field).__name__)
        return self.mean_field

    def _extract_mo_coeffs(self) -> None:
        """Extract and store occupied MO coefficients per k-point after SCF.

        This method is necessary for PBC but not OBC due to how PySCF stores
        MO coefficients:

        - **OBC**: ``mean_field.mo_coeff`` is a single ``(nao, nmo)`` array that
          can be used directly in matrix multiplication.

        - **PBC**: ``mean_field.mo_coeff`` is a list of ``nk`` arrays, where each
          k-point's array has shape ``(nao, nmo_k)``. The number of occupied
          orbitals ``nocc_k`` can vary across k-points due to band filling at
          different points in the Brillouin zone.

        This method:

        1. Iterates over k-points
        2. Identifies occupied orbitals using occupation threshold (>0.9 for
           occupied, >1.1 for doubly occupied in restricted case)
        3. Extracts only the occupied columns from each k-point's MO matrix
        4. Stores results in ``self._mo_coeff = (alpha_list, beta_list)``

        Pre-extraction is done for performance (avoid repeated slicing) and to
        handle the variable-size arrays that cannot be directly stacked.
        """
        nkpts = len(self.kpts)
        alpha_coeffs: list[jnp.ndarray] = []
        beta_coeffs: list[jnp.ndarray] = []

        for k in range(nkpts):
            if self.restricted:
                occ_alpha = self.mean_field.mo_occ[k] > 0.9
                occ_beta = self.mean_field.mo_occ[k] > 1.1
                alpha_coeffs.append(
                    jnp.asarray(self.mean_field.mo_coeff[k][:, occ_alpha])
                )
                beta_coeffs.append(
                    jnp.asarray(self.mean_field.mo_coeff[k][:, occ_beta])
                )
            else:
                occ_alpha = self.mean_field.mo_occ[0][k] > 0.9
                occ_beta = self.mean_field.mo_occ[1][k] > 0.9
                alpha_coeffs.append(
                    jnp.asarray(self.mean_field.mo_coeff[0][k][:, occ_alpha])
                )
                beta_coeffs.append(
                    jnp.asarray(self.mean_field.mo_coeff[1][k][:, occ_beta])
                )

        self._mo_coeff = (alpha_coeffs, beta_coeffs)

    def eval_mos(self, positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Evaluate MOs at positions using all k-points.

        For periodic systems, the molecular orbital at k-point :math:`\mathbf{k}`
        is constructed from Bloch-summed atomic orbitals:

        .. math::

            \psi_{n\mathbf{k}}(\mathbf{r}) = \sum_\mu C_{\mu n}^{\mathbf{k}}
                \phi_{\mu\mathbf{k}}(\mathbf{r})

        where :math:`C_{\mu n}^{\mathbf{k}}` are the MO coefficients and
        :math:`\phi_{\mu\mathbf{k}}` are Bloch AOs.

        When an electron coordinate :math:`\mathbf{r}` lies outside the primitive
        cell, it is wrapped back: :math:`\mathbf{r} = \mathbf{r}' + \mathbf{R}`
        where :math:`\mathbf{r}'` is in the cell and :math:`\mathbf{R}` is a
        lattice vector. Due to Bloch's theorem, this introduces a phase factor:

        .. math::

            \psi_{n\mathbf{k}}(\mathbf{r}) =
                e^{i\mathbf{k}\cdot\mathbf{R}} \psi_{n\mathbf{k}}(\mathbf{r}')

        Args:
            positions: Electron coordinates with shape ``(N, 3)``.

        Returns:
            Tuple of ``(alpha_mos, beta_mos)``, each with shape
            ``(nk, N, max_occ)`` where ``max_occ`` is the maximum number of
            occupied orbitals across k-points. Zero-padded for k-points with
            fewer occupied orbitals.

        Raises:
            RuntimeError: If Hartree-Fock calculation has not been performed.
        """
        if self._mo_coeff is None:
            raise RuntimeError("Mean-field calculation has not been run.")

        # Wrap coordinates to primitive cell: r = r' + R where R is the displacement
        latvec = jnp.asarray(self._cell.a)
        wrapped = wrap_positions(positions, latvec)

        # Evaluate Bloch AOs at wrapped coordinates: shape (nk, N, nao)
        aos = self.eval_aos(wrapped, self.kpts)

        # Apply Bloch phase correction: exp(i k · R)
        R = positions - wrapped
        kdotR = jnp.einsum("ki,ni->kn", self.kpts, R)
        aos = aos * jnp.exp(1j * kdotR)[..., None]

        # Contract with MO coefficients per k-point.
        # Pad to common size since different k-points may have different occupancy
        # (band filling varies across the Brillouin zone).
        nkpts = len(self.kpts)
        npos = positions.shape[0]
        alpha_coeffs = self._mo_coeff[0]
        beta_coeffs = self._mo_coeff[1]
        max_alpha = max(c.shape[1] for c in alpha_coeffs) if alpha_coeffs else 0
        max_beta = max(c.shape[1] for c in beta_coeffs) if beta_coeffs else 0

        alpha_list = []
        beta_list = []
        for k in range(nkpts):
            alpha_k = aos[k] @ alpha_coeffs[k]  # (N, nocc_alpha_k)
            beta_k = aos[k] @ beta_coeffs[k]  # (N, nocc_beta_k)
            # Pad to max size
            alpha_pad = max_alpha - alpha_k.shape[1]
            beta_pad = max_beta - beta_k.shape[1]
            if alpha_pad > 0:
                alpha_k = jnp.pad(alpha_k, ((0, 0), (0, alpha_pad)))
            if beta_pad > 0:
                beta_k = jnp.pad(beta_k, ((0, 0), (0, beta_pad)))
            alpha_list.append(alpha_k)
            beta_list.append(beta_k)

        alpha_mos = jnp.stack(alpha_list) if alpha_list else jnp.zeros((nkpts, npos, 0))
        beta_mos = jnp.stack(beta_list) if beta_list else jnp.zeros((nkpts, npos, 0))

        return alpha_mos, beta_mos

    def eval_orbitals(
        self, pos: jnp.ndarray, nspins: tuple[int, int]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Evaluate occupied orbital matrices for the Slater determinant.

        This method produces dense orbital matrices by evaluating all electrons
        at all k-points and concatenating the results. The k-point structure is
        implicit in the MO coefficients from the periodic HF calculation.

        For each spin, the orbital matrix has shape ``(n_spin, n_spin)`` where:

        - Row i corresponds to electron i
        - Column j corresponds to orbital j (ordered by k-point)

        The resulting matrices can be used directly with ``jnp.linalg.slogdet``
        to compute the Slater determinant. The block-diagonal structure (electrons
        at different k-points being orthogonal) emerges naturally from the physics.

        Args:
            pos: Electron positions with shape ``(..., nelec, 3)``.
            nspins: Tuple of ``(n_alpha, n_beta)`` electrons.

        Returns:
            Tuple of ``(alpha_matrix, beta_matrix)`` orbital matrices with
            shapes ``(..., n_alpha, n_alpha)`` and ``(..., n_beta, n_beta)``.

        Raises:
            RuntimeError: If Hartree-Fock calculation has not been performed.
        """
        if self._mo_coeff is None:
            raise RuntimeError("Mean-field calculation has not been run.")

        n_alpha, n_beta = nspins
        leading_dims = pos.shape[:-2]
        pos_flat = jnp.reshape(pos, [-1, 3])  # (batch*nelec, 3)

        # Wrap coordinates to primitive cell and compute Bloch phase correction
        latvec = jnp.asarray(self._cell.a)
        wrapped = wrap_positions(pos_flat, latvec)

        # Evaluate Bloch AOs at all k-points: shape (nk, batch*nelec, nao)
        aos = self.eval_aos(wrapped, self.kpts)

        # Apply Bloch phase: exp(i k · R) where R = pos_flat - wrapped
        R = pos_flat - wrapped
        kdotR = jnp.einsum("ki,ni->kn", self.kpts, R)
        aos = aos * jnp.exp(1j * kdotR)[..., None]

        # Reshape AOs to separate electrons: (nk, batch, nelec, nao)
        nelec = n_alpha + n_beta
        nkpts = len(self.kpts)
        aos = jnp.reshape(aos, (nkpts, -1, nelec, aos.shape[-1]))

        # For each spin, iterate over k-points and contract with MO coefficients
        def build_orbital_matrix(
            aos: jnp.ndarray, electron_slice: slice, mo_coeffs: list
        ) -> jnp.ndarray:
            """Returns contracted AOs with MO coefficients for one spin."""
            # aos shape: (nk, batch, nelec, nao)
            # Select electrons for this spin: (nk, batch, n_spin, nao)
            aos_spin = aos[:, :, electron_slice, :]
            # Contract each k-point and concatenate
            mo_list = [aos_spin[k] @ mo_coeffs[k] for k in range(nkpts)]
            # Concatenate along orbital axis: (batch, n_spin, total_orbs)
            return jnp.concatenate(mo_list, axis=-1)

        alpha_mos = build_orbital_matrix(aos, slice(0, n_alpha), self._mo_coeff[0])
        beta_mos = build_orbital_matrix(aos, slice(n_alpha, nelec), self._mo_coeff[1])

        # Reshape to output format: (..., n_spin, n_spin)
        # Handle zero-electron cases explicitly to avoid reshape issues with -1
        if n_alpha > 0:
            alpha_matrix = jnp.reshape(alpha_mos, (*leading_dims, n_alpha, -1))
            alpha_matrix = alpha_matrix[..., :n_alpha]
        else:
            alpha_matrix = jnp.zeros((*leading_dims, 0, 0), dtype=alpha_mos.dtype)

        if n_beta > 0:
            beta_matrix = jnp.reshape(beta_mos, (*leading_dims, n_beta, -1))
            beta_matrix = beta_matrix[..., :n_beta]
        else:
            beta_matrix = jnp.zeros((*leading_dims, 0, 0), dtype=beta_mos.dtype)

        return alpha_matrix, beta_matrix

    def eval_slater(self, pos: jnp.ndarray, nspins: tuple[int, int]) -> jnp.ndarray:
        """Evaluate Slater determinant sign and log|det|.

        Args:
            pos: An array of electron positions to evaluate the orbitals at.
            nspins: Tuple with number of spin up and spin down electrons.

        Returns:
            Log value of Slater determinant (complex).
        """
        alpha_matrix, beta_matrix = self.eval_orbitals(pos, nspins)
        sign, logdet = _eval_slater_from_orbitals(alpha_matrix, beta_matrix)
        return logdet + jnp.log(sign)

    def get_orbital_kpoints(self) -> jnp.ndarray:
        """Get k-point assignment for each orbital in the wavefunction.

        In periodic systems, each orbital is associated with a specific k-point.
        This method constructs an array where each row is the k-point vector
        for the corresponding orbital.

        The orbitals are ordered as: all spin-up orbitals (concatenated across
        k-points), then all spin-down orbitals.

        Returns:
            Array of shape ``(n_orbitals, 3)`` where ``n_orbitals`` is the total
            number of occupied orbitals across all k-points and spins. Each row
            contains the k-point coordinates for that orbital.

        Raises:
            RuntimeError: If Hartree-Fock calculation has not been performed.

        Example:
            For a system with 4 k-points and 1 occupied orbital per k-point per
            spin, the returned array has shape ``(8, 3)`` with k-points ordered
            as ``[k0, k1, k2, k3, k0, k1, k2, k3]``.
        """
        if self._mo_coeff is None:
            raise RuntimeError("Mean-field calculation has not been run.")

        alpha_coeffs, beta_coeffs = self._mo_coeff
        kpts = self.kpts

        # Count occupied orbitals per k-point for each spin
        # coeff.shape[1] is the number of occupied orbitals at that k-point
        alpha_counts = jnp.array([c.shape[1] for c in alpha_coeffs])
        beta_counts = jnp.array([c.shape[1] for c in beta_coeffs])

        # Build k-point indices by repeating each k-index by its orbital count
        # e.g., counts=[1,2,1] -> indices=[0,1,1,2]
        k_indices = jnp.arange(len(kpts))
        alpha_indices = jnp.repeat(k_indices, alpha_counts)
        beta_indices = jnp.repeat(k_indices, beta_counts)

        return jnp.concatenate([kpts[alpha_indices], kpts[beta_indices]])
