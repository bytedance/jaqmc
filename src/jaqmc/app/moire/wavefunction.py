# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import field
from typing import Literal

import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp

from jaqmc.array_types import Params
from jaqmc.geometry.pbc import DistanceType, SymmetryType
from jaqmc.utils.supercell import (
    get_reciprocal_vectors,
    get_supercell_copies,
    get_supercell_kpts_in_first_bz,
)
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction import Wavefunction
from jaqmc.wavefunction.backbone.ferminet import FermiLayers
from jaqmc.wavefunction.input.atomic import SolidFeatures
from jaqmc.wavefunction.output.logdet import ComplexLogDetOutput, LogDet
from jaqmc.wavefunction.output.orbital import SplitChannelDense

from .config import MoireConfig
from .data import MoireData

__all__ = ["MoireWavefunction"]


def _pseudospin_phase_exponents(nlayers: int) -> tuple[int, ...]:
    r"""Returns the symmetric pseudospin phase exponents for ``nlayers`` layers.

    Each layer orbital :math:`\phi_j` is dressed with :math:`e^{i m_j s}` in the
    identity determinant. The exponents :math:`m_j` are the layer-pseudospin
    projections, symmetric about zero and ordered from high to low:

    - even ``nlayers=2k``: ``(k, ..., 1, -1, ..., -k)``, skipping zero. The
      bilayer case ``nlayers=2`` recovers ``(1, -1)``, i.e. ``e^{is}, e^{-is}``.
    - odd ``nlayers=2k+1``: ``(k, ..., 1, 0, -1, ..., -k)``, including zero. The
      trilayer case ``nlayers=3`` gives ``(1, 0, -1)``, i.e. ``e^{is}, 1,
      e^{-is}``.

    Args:
        nlayers: Number of moire layers.

    Returns:
        Pseudospin phase exponents, one per layer, ordered high to low.
    """
    if nlayers % 2 == 0:
        half = nlayers // 2
        return tuple(m for m in range(half, -half - 1, -1) if m != 0)
    half = nlayers // 2
    return tuple(range(half, -half - 1, -1))


def _combine_translation_and_det_axes(orbitals: jnp.ndarray) -> jnp.ndarray:
    """Folds a leading translation axis into the determinant axis.

    Args:
        orbitals: Orbital tensor with a leading translation axis.

    Returns:
        Orbital tensor with translation sectors folded into its determinant axis.
    """
    n_translation = orbitals.shape[0]
    orbitals = jnp.moveaxis(orbitals, 0, orbitals.ndim - 4)
    ndets, nelec = orbitals.shape[-3:-1]
    return orbitals.reshape(*orbitals.shape[:-4], n_translation * ndets, nelec, nelec)


class MultiPhaseLayer(nn.Module):
    r"""Apply k-list-dependent phase dressing to orbital matrices.

    Each orbital is dressed with a trainable plane-wave envelope over the
    k-point list,

    .. math::

        \sum_j \pi_{ij}\, e^{i\mathbf k_j\cdot\mathbf r},

    where :math:`\mathbf k_j` runs over the entries of ``klist`` and
    :math:`\pi_{ij}` are the trainable ``multik`` parameters.

    The center-of-mass momentum projection over supercell copies
    :math:`\mathbf l` is applied outside this layer, in
    ``_get_momentum_projected_multiphase_orbitals``.

    Args:
        klist: Cartesian k-point list used in the multiphase expansion.
    """

    klist: jnp.ndarray

    @nn.compact
    def __call__(
        self,
        *,
        orbital: jnp.ndarray,
        positions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Applies the multiphase factor to an orbital tensor.

        Args:
            orbital: Complex orbital tensor with shape
                ``(..., ndets, nelec, nelec)``.
            positions: Electron positions with shape ``(..., nelec, 2)``.

        Returns:
            Complex orbital tensor with the same shape as ``orbital``.
        """
        nelec = orbital.shape[-1]
        ndets = orbital.shape[-3]
        klist = jnp.asarray(self.klist)
        nk = int(klist.shape[0])

        def init_multik(rng, shape):
            del shape
            # multik ~ normal / sqrt(nk), repeated across each electron's det axis.
            base = jax.random.normal(rng, (nk, nelec), dtype=klist.dtype) / jnp.sqrt(
                float(nk)
            )
            return jnp.repeat(base, ndets, axis=-1)

        multik = self.param("multik", init_multik, (nk, nelec * ndets))
        kdot = positions @ klist.T
        trig_features = jnp.stack([jnp.cos(kdot), jnp.sin(kdot)], axis=-2)
        mixed = jnp.dot(trig_features, multik)
        cos_mix = mixed[..., 0, :]
        sin_mix = mixed[..., 1, :]
        phase = (cos_mix + 1j * sin_mix).reshape(*cos_mix.shape[:-1], nelec, ndets)
        # (..., nelec_row, nelec_col, ndets) -> (..., ndets, nelec_row, nelec_col)
        phase = jnp.moveaxis(phase, -1, -3)
        return orbital * phase.reshape((1,) * (orbital.ndim - phase.ndim) + phase.shape)


class MoireWavefunction(Wavefunction[MoireData, ComplexLogDetOutput]):
    r"""Wavefunction ansatz for multilayer moire systems.

    The network produces ``nlayers`` orbital matrices :math:`\phi_1, \dots,
    \phi_{n_{\rm layers}}` in the layer-pseudospin basis, one spatial orbital per
    moire layer.  The wavefunction amplitude sums these with symmetric
    pseudospin phases :math:`e^{i m_j s_i}`,

    .. math::

        \Phi_0(\mathbf r_i, s_i) = \sum_j \phi_j(\mathbf r_i) e^{i m_j s_i},

    where the exponents :math:`m_j` are symmetric about zero (see
    :func:`_pseudospin_phase_exponents`): an even ``nlayers`` skips the zero
    exponent (bilayer ``nlayers=2`` gives :math:`e^{\pm i s}`), while an odd
    ``nlayers`` includes it (trilayer ``nlayers=3`` gives
    :math:`e^{i s}, 1, e^{-i s}`).

    Args:
        nspins: Tuple of (n_up, n_down) electrons.
        singlephase_klists: Spin-channel k-point lists for integer CI
            singlephase Bloch factors.
        multiphase_klist: K-point list for fractional FCI multiphase dressing.
        translation_vectors: Supercell translation vectors used by FCI
            center-of-mass momentum projection.
        simulation_lattice: Lattice vectors of the simulation cell.
        primitive_lattice: Lattice vectors of the primitive computational cell.
        twist: Twist in simulation-cell reciprocal fractional coordinates.
        phase_mode: ``"singlephase"`` for integer CI or ``"multiphase"`` for
            fractional FCI.
        k_com: Center-of-mass momentum sector used for FCI momentum projection,
            in simulation-cell reciprocal fractional coordinates. Multiphase
            wavefunctions require a concrete sector; phase-input configuration
            substitutes the default ``(0.0, 0.0)`` Gamma sector when unset.
            Singlephase wavefunctions do not support this argument.
        hidden_dims_single: Hidden dimensions for single-electron streams.
        hidden_dims_double: Hidden dimensions for pairwise streams.
        ndets: Number of determinants before center-of-mass momentum projection.
        nlayers: Number of moire layers, i.e. the number of spatial orbitals
            summed with symmetric layer phases. Defaults to ``2``.
        distance_type: Method to compute periodic distances.
        sym_type: Symmetry type for features.
    """

    nspins: tuple[int, int] = runtime_dep()
    singlephase_klists: tuple[jnp.ndarray, jnp.ndarray] | None = runtime_dep(
        default=None
    )
    multiphase_klist: jnp.ndarray | None = runtime_dep(default=None)
    translation_vectors: jnp.ndarray | None = runtime_dep(default=None)
    simulation_lattice: jnp.ndarray = runtime_dep()
    primitive_lattice: jnp.ndarray = runtime_dep()
    twist: jnp.ndarray | None = runtime_dep(default=None)
    phase_mode: Literal["singlephase", "multiphase"] = runtime_dep(
        default="singlephase"
    )
    k_com: tuple[float, float] | None = None
    hidden_dims_single: list[int] = field(default_factory=lambda: [256] * 4)
    hidden_dims_double: list[int] = field(default_factory=lambda: [32] * 4)
    ndets: int = 1
    nlayers: int = 2
    distance_type: DistanceType = DistanceType.tri
    sym_type: SymmetryType = SymmetryType.minimal

    def setup(self) -> None:
        """Validates the phase mode and builds the wavefunction layers.

        Raises:
            ValueError: If the selected phase mode lacks or forbids inputs for
                center-of-mass momentum projection.
        """
        if self.phase_mode == "singlephase":
            if self.k_com is not None:
                raise ValueError("k_com must be None when phase_mode='singlephase'")
        else:
            if self.k_com is None:
                raise ValueError("k_com must be provided when phase_mode='multiphase'")
            if self.multiphase_klist is None:
                raise ValueError(
                    "multiphase_klist must be provided when phase_mode='multiphase'"
                )
            if self.translation_vectors is None:
                raise ValueError(
                    "translation_vectors must be provided when phase_mode='multiphase'"
                )
            self.multiphase_layer = MultiPhaseLayer(
                klist=self.multiphase_klist,
                name="multiphase_layer",
            )
        self.feature_layer = SolidFeatures(
            simulation_lattice=self.simulation_lattice,
            primitive_lattice=self.primitive_lattice,
            ae_lattice=(
                "primitive" if self.phase_mode == "singlephase" else "simulation"
            ),
            distance_type=self.distance_type,
            sym_type=self.sym_type,
        )
        hidden_dims = list(zip(self.hidden_dims_single, self.hidden_dims_double))
        self.backbone_layer = FermiLayers(self.nspins, hidden_dims)
        # One (real, imag) orbital head per moire layer. Preserve the historical
        # 1-based ``{real,imag}_orbital_layer_{n}`` outer parameter names.
        orbital_features = [self.ndets, sum(self.nspins)]
        self.real_orbital_layers = tuple(
            SplitChannelDense(
                channels=self.nspins,
                features=orbital_features,
                use_bias=False,
                name=f"real_orbital_layer_{layer_idx + 1}",
            )
            for layer_idx in range(self.nlayers)
        )
        self.imag_orbital_layers = tuple(
            SplitChannelDense(
                channels=self.nspins,
                features=orbital_features,
                use_bias=False,
                name=f"imag_orbital_layer_{layer_idx + 1}",
            )
            for layer_idx in range(self.nlayers)
        )
        self.logdet_layer = LogDet()

    def configure_phase_inputs(self, system_config: MoireConfig) -> None:
        r"""Derive phase inputs from the resolved moire system.

        Set the phase mode, k-points, translations, and default COM momentum.

        Args:
            system_config: Resolved moire system configuration providing filling
                and supercell geometry inputs.

        Raises:
            ValueError: If singlephase integer filling is incompatible with the
                configured spin channels.
        """
        lattice_area = abs(np.linalg.det(np.asarray(system_config.lattice_vectors)))
        moire_area = abs(np.linalg.det(np.asarray(system_config.moire_lattice_vectors)))
        filling = system_config.nelec / (
            system_config.scale * lattice_area / moire_area
        )
        self.phase_mode = (
            "singlephase" if math.isclose(filling, round(filling)) else "multiphase"
        )

        S = jnp.asarray(system_config.supercell_matrix)
        nk = system_config.scale
        kpts = get_supercell_kpts_in_first_bz(S, self.primitive_lattice)
        if self.phase_mode == "singlephase":
            if any(nspin % nk != 0 for nspin in self.nspins):
                raise ValueError(
                    "singlephase integer filling requires each spin channel to be "
                    f"divisible by nk={nk}, got electron_spins={self.nspins}."
                )
            original_nspins = tuple(nspin // nk for nspin in self.nspins)
            # Each spin channel repeats the folded kmesh once per primitive electron.
            self.singlephase_klists = (
                jnp.tile(kpts, (original_nspins[0], 1)),
                jnp.tile(kpts, (original_nspins[1], 1)),
            )
        else:
            self.multiphase_klist = kpts
            # Primitive-cell copies l_a enumerated inside the supercell from S.
            self.translation_vectors = get_supercell_copies(self.primitive_lattice, S)
            # Fractional filling: an unset k_com means "auto-select the default
            # sector", which is the Gamma point (k_com=0). Translation projection
            # is therefore on by default; users pick another sector via wf.k_com
            # in YAML/CLI. (Leaving k_com=None on the raw MoireWavefunction
            # instead disables projection, but the workflow never surfaces that.)
            if self.k_com is None:
                self.k_com = (0.0, 0.0)

    # Shared dense orbital core.
    def _get_orbital_blocks(self, data: MoireData) -> tuple[jnp.ndarray, ...]:
        """Computes one dense complex orbital matrix per moire layer.

        Args:
            data: The input data containing electron positions and spin angles.

        Returns:
            ``nlayers`` dense matrices with shape ``(ndets, nelec, nelec)``.
        """
        embedding = self.feature_layer(data.positions)
        h_one, _ = self.backbone_layer(
            embedding["ae_features"], embedding["ee_features"]
        )
        return tuple(
            jnp.transpose(real_layer(h_one) + 1j * imag_layer(h_one), (1, 0, 2))
            for real_layer, imag_layer in zip(
                self.real_orbital_layers, self.imag_orbital_layers
            )
        )

    def _apply_twist(
        self,
        orbital: jnp.ndarray,
        positions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Applies the twist Bloch phase to all dense orbital rows.

        Args:
            orbital: Dense orbital matrix with shape ``(..., ndets, nelec, nelec)``.
            positions: Electron positions with shape ``(..., nelec, 2)``.

        Returns:
            Dense orbital matrix with the same shape as ``orbital``.
        """
        if self.twist is None:
            return orbital
        k_twist = jnp.asarray(self.twist) @ get_reciprocal_vectors(
            self.simulation_lattice
        )
        row_phase = jnp.exp(1j * (positions @ k_twist))[..., None, :, None]
        return orbital * row_phase

    # Phase-mode layer-orbital pipelines.
    def _get_singlephase_layer_orbitals(
        self, data: MoireData
    ) -> tuple[jnp.ndarray, ...]:
        """Builds singlephase orbital matrices before spin-angle projection.

        Args:
            data: Electron positions and spin angles.

        Returns:
            One layer-pseudospin orbital matrix per moire layer.
        """
        positions = data.positions
        layer_orbitals = self._get_orbital_blocks(data)
        assert self.singlephase_klists is not None
        klist = jnp.concatenate(
            tuple(
                self.singlephase_klists[spin_idx]
                for spin_idx, nspin in enumerate(self.nspins)
                if nspin > 0
            ),
            axis=0,
        )
        phase = jnp.exp(1j * positions @ klist.T)[..., None, :, :]
        return tuple(
            self._apply_twist(orbital * phase, positions) for orbital in layer_orbitals
        )

    def _get_momentum_projected_multiphase_orbitals(
        self, data: MoireData
    ) -> tuple[jnp.ndarray, ...]:
        """Builds momentum-resolved multiphase orbital matrices.

        Args:
            data: Electron positions and spin angles.

        Returns:
            One momentum-resolved layer-pseudospin matrix per moire layer, each
            with translation and determinant axes combined.
        """
        positions = data.positions
        translation_vectors = jnp.asarray(self.translation_vectors)
        k_com_cart = jnp.asarray(self.k_com) @ get_reciprocal_vectors(
            self.simulation_lattice
        )
        translation_phases = jnp.exp(
            1j * (translation_vectors @ k_com_cart) / sum(self.nspins)
        )
        shifted_shape = (
            (translation_vectors.shape[0],) + (1,) * (positions.ndim - 1) + (2,)
        )
        shifted_positions = positions[None, ...] - translation_vectors.reshape(
            shifted_shape
        )

        def evaluate_shifted(
            shifted_position: jnp.ndarray,
        ) -> tuple[jnp.ndarray, ...]:
            shifted_data = data.merge({"positions": shifted_position})
            return self._get_orbital_blocks(shifted_data)

        layer_orbitals = jax.vmap(evaluate_shifted)(shifted_positions)
        momentum_projected = tuple(
            translation_phases.astype(orbital.dtype).reshape(
                (translation_phases.shape[0],) + (1,) * (orbital.ndim - 1)
            )
            * orbital
            for orbital in layer_orbitals
        )
        layer_stack = jnp.stack(momentum_projected, axis=0)
        multiphase = self.multiphase_layer(
            orbital=layer_stack,
            positions=shifted_positions,
        )
        multiphase = self._apply_twist(multiphase, shifted_positions)
        return tuple(
            _combine_translation_and_det_axes(multiphase[layer_idx])
            for layer_idx in range(self.nlayers)
        )

    def get_layer_components(
        self, data: MoireData
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r"""Computes layer orbitals, phase amplitudes, and identity orbitals.

        Args:
            data: The input data containing electron positions and layer angles.

        Returns:
            ``(phi, chi, phi0)``, with layer orbitals ``phi`` shaped
            ``(nlayers, ndets, nelec, nelec)``, phase amplitudes ``chi`` shaped
            ``(nelec, nlayers)``, and identity orbitals ``phi0`` shaped
            ``(ndets, nelec, nelec)``. The identity contract is
            :math:`\Phi^0_{dij}=\sum_l\chi_{il}\phi_{ldij}`.
        """
        if self.phase_mode == "singlephase":
            layer_orbitals = self._get_singlephase_layer_orbitals(data)
        else:
            layer_orbitals = self._get_momentum_projected_multiphase_orbitals(data)
        phi = jnp.stack(layer_orbitals, axis=0)
        chi = jnp.stack(
            [
                jnp.exp(1j * m * data.spin_coords)
                for m in _pseudospin_phase_exponents(self.nlayers)
            ],
            axis=-1,
        )
        phi0 = jnp.einsum("il,ldij->dij", chi, phi)
        return phi, chi, phi0

    def get_orbitals(self, data: MoireData) -> jnp.ndarray:
        """Computes the identity orbital matrix for the given data.

        Args:
            data: The input data containing electron positions and layer angles.

        Returns:
            Complex orbital matrix with shape ``(ndets, nelec, nelec)``.
        """
        return self.get_layer_components(data)[2]

    def __call__(self, data: MoireData) -> ComplexLogDetOutput:
        """Evaluates the wavefunction for the given data.

        Args:
            data: The input data containing electron positions and spin angles.

        Returns:
            A dictionary containing the log of the wavefunction amplitude
            (``logpsi``) and other outputs from ``LogDet``.
        """
        return self.logdet_layer(self.get_orbitals(data))

    # Bound evaluation API.
    def layer_components(
        self, params: Params, data: MoireData
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluates layer orbitals, phase amplitudes, and identity orbitals.

        Args:
            params: The wavefunction parameters.
            data: The input data containing electron positions and layer angles.

        Returns:
            Tuple ``(phi, chi, phi0)`` from :meth:`get_layer_components`.
        """
        return self.apply(params, data, method=self.get_layer_components)  # type: ignore

    def orbitals(self, params: Params, data: MoireData) -> jnp.ndarray:
        """Evaluates the orbital matrix for the given parameters and data.

        Args:
            params: The wavefunction parameters.
            data: The input data containing electron positions and spin angles.

        Returns:
            Complex orbital matrix with shape ``(..., ndets, nelec_total,
            nelec_total)``. For momentum-resolved multiphase orbitals, the
            determinant axis also includes translation sectors.
        """
        return self.apply(params, data, method=self.get_orbitals)  # type: ignore

    def logpsi(self, params: Params, data: MoireData) -> jnp.ndarray:
        """Evaluates the log wavefunction amplitude.

        Args:
            params: The wavefunction parameters.
            data: The input data containing electron positions and spin angles.

        Returns:
            Complex logarithm of the wavefunction amplitude.
        """
        return self.evaluate(params, data)["logpsi"]

    def phase_logpsi(
        self, params: Params, data: MoireData
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluates the wavefunction phase and log amplitude.

        Args:
            params: The wavefunction parameters.
            data: The input data containing electron positions and spin angles.

        Returns:
            A tuple ``(phase, log_abs)`` where ``phase`` is ``exp(1j * angle)``
            and ``log_abs`` is the real part of ``logpsi``.
        """
        logpsi = self.logpsi(params, data)
        return jnp.exp(1j * logpsi.imag), logpsi.real
