# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Nonlocal ECP integration using spherical quadrature.

.. seealso:: :doc:`/guide/estimators/ecp` for formulas and background.
"""

from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey
from jaqmc.geometry.pbc import wrap_positions

from .quadrature import Quadrature

__all__ = ["make_nonlocal_integral"]


def make_nonlocal_integral(
    num_channels: int,
    quadrature: Quadrature,
    lattice: jnp.ndarray | None = None,
    twist: jnp.ndarray | None = None,
) -> Callable[..., jnp.ndarray]:
    """Create a nonlocal ECP integral evaluator.

    Args:
        num_channels: Number of ECP channels (1 local + N nonlocal).
        quadrature: Quadrature rule for angular integration.
        lattice: Lattice vectors for PBC, shape (3, 3). If None, assumes OBC.
        twist: Twist angles in fractional coordinates for Bloch phase, shape (3,).
            Only used when lattice is provided.

    Returns:
        Callable that computes per-channel nonlocal integrals.
    """
    n_nonlocal = num_channels - 1

    def evaluate(
        phase_wf: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
        electrons: jnp.ndarray,
        atom_positions: jnp.ndarray,
        key: PRNGKey,
    ) -> jnp.ndarray:
        r"""Compute nonlocal ECP integrals.

        Uses scan over electrons for memory efficiency, with vmap over
        atoms and quadrature points for parallelism.

        Args:
            phase_wf: Wavefunction returning (phase, :math:`\log|\psi|`) for
                electron positions of shape (n_elec, 3).
            electrons: Electron positions, shape (n_elec, 3).
            atom_positions: Nearest atom image for each electron-atom pair,
                shape (n_elec, n_atoms, 3).
            key: PRNG key for random rotations.

        Returns:
            Per-channel integrals, shape (n_elec, n_atoms, n_nonlocal).
            Multiply by radial potentials ``V_l(r)`` and sum to get energy.
        """
        n_elec = electrons.shape[0]
        n_atoms = atom_positions.shape[1]

        if n_atoms == 0 or n_nonlocal == 0:
            return jnp.zeros((n_elec, n_atoms, n_nonlocal))

        # Reference wavefunction value
        ref_phase, ref_log_psi = phase_wf(electrons)

        # Rotated quadrature points for each electron
        quad_points = quadrature.sample_rotated_points(n_elec, key)
        n_quad = quad_points.shape[1]

        @partial(jax.vmap, in_axes=0)
        def wf_ratio(displaced_electrons: jnp.ndarray) -> jnp.ndarray:
            r"""Compute wavefunction ratio for displaced configuration.

            Args:
                displaced_electrons: Full electron configuration, (n_elec, 3).

            Returns:
                Ratio :math:`\psi(\mathbf{r}')/\psi(\mathbf{r})`.
            """
            phase, log_psi = phase_wf(displaced_electrons)
            return phase * jnp.conj(ref_phase) * jnp.exp(log_psi - ref_log_psi)

        def electron_contribution(carry, x):
            """Compute integral for one electron-atom pair.

            Args:
                carry: Unused (scan requires a carry).
                x: Tuple of (elec_idx, atom_pos, elec_quad_points).

            Returns:
                (carry, channel_integrals) with shape (n_nonlocal,).
            """
            elec_idx, atom_pos, elec_quad_points = x

            # Compute atom-electron geometry
            electron = electrons[elec_idx]
            r_ae_vec = electron - atom_pos
            r_ae_dist = jnp.linalg.norm(r_ae_vec)
            r_ae_dir = r_ae_vec / r_ae_dist

            # Displaced positions on sphere around atom
            displaced_pos = atom_pos + r_ae_dist * elec_quad_points  # (n_quad, 3)

            # Build full electron configurations with this electron displaced
            displaced_electrons = jnp.tile(electrons, (n_quad, 1, 1))
            displaced_electrons = displaced_electrons.at[:, elec_idx, :].set(
                displaced_pos
            )

            # Compute wavefunction ratios
            if lattice is not None:
                wrapped_pos, bloch_phase = _wrap_with_bloch_phase(
                    displaced_pos, lattice, twist
                )
                displaced_electrons = displaced_electrons.at[:, elec_idx, :].set(
                    wrapped_pos
                )
                ratios = wf_ratio(displaced_electrons) * bloch_phase
            else:
                ratios = wf_ratio(displaced_electrons)

            # Legendre polynomials at cos(theta)
            cos_theta = jnp.dot(elec_quad_points, r_ae_dir)
            pl_values = legendre_polynomials(cos_theta, n_nonlocal)

            # Integrate each angular momentum channel
            integrals = []
            for angular_l in range(n_nonlocal):
                integrand = pl_values[angular_l] * ratios
                integral = quadrature.integrate(integrand) * (2 * angular_l + 1)
                integrals.append(integral)

            channel_integrals = jnp.stack(integrals) / (4 * jnp.pi)  # (n_nonlocal,)
            return carry, channel_integrals

        @partial(jax.vmap, in_axes=1, out_axes=1)
        def per_atom_integral(atom_pos_per_elec: jnp.ndarray) -> jnp.ndarray:
            """Compute integrals for one atom across all electrons.

            Uses scan over electrons to limit memory usage.

            Args:
                atom_pos_per_elec: This atom's position for each electron, (n_elec, 3).

            Returns:
                Integrals for this atom, shape (n_elec, n_nonlocal).
            """
            xs = (jnp.arange(n_elec), atom_pos_per_elec, quad_points)
            _, integrals = jax.lax.scan(electron_contribution, None, xs)
            return integrals  # (n_elec, n_nonlocal)

        return per_atom_integral(atom_positions)

    return evaluate


def _wrap_with_bloch_phase(
    positions: jnp.ndarray,
    lattice: jnp.ndarray,
    twist: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Wrap positions into primary cell and compute Bloch phase.

    Returns:
        (wrapped_positions, bloch_phase) tuple.
    """
    wrapped_positions = wrap_positions(positions, lattice)
    if twist is not None:
        lattice_shift = positions - wrapped_positions
        fractional_shift = lattice_shift @ jnp.linalg.inv(lattice)
        kdot = 2 * jnp.pi * jnp.dot(fractional_shift, jnp.mod(twist, 1.0))
        bloch_phase = jnp.exp(1j * kdot)
    else:
        bloch_phase = jnp.ones(positions.shape[:-1], dtype=positions.dtype)
    return wrapped_positions, bloch_phase


def legendre_polynomials(x: jnp.ndarray, num_l: int) -> jnp.ndarray:
    r"""Evaluate Legendre polynomials :math:`P_0(x), P_1(x), \ldots, P_{n-1}(x)`.

    Args:
        x: Evaluation points (typically :math:`\cos\theta`).
        num_l: Number of Legendre polynomials to evaluate.

    Returns:
        Array of shape (num_l, *x.shape).

    Raises:
        ValueError: If ``num_l`` exceeds the number of hardcoded polynomials.
    """
    if num_l > 3:
        raise ValueError(
            f"Legendre polynomials up to l=2 are supported, "
            f"but num_l={num_l} was requested."
        )
    if num_l <= 0:
        return jnp.zeros((0, *x.shape), dtype=x.dtype)

    all_polys = [
        jnp.ones_like(x),
        x,
        (3 * x**2 - 1) / 2,
    ]
    return jnp.stack(all_polys[:num_l], axis=0)
