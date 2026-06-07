# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""End-to-end correctness tests for ``PHEnergy``.

The PH paper formulas are taken as given and audited by code review of
:mod:`jaqmc.estimator.ph._standard` and
:mod:`jaqmc.estimator.ph._forward_laplacian` (both cite Bennett et al.,
2022). This file pins the remaining concerns:

- ``test_ph_forward_laplacian_total_energy_matches_standard_backend``:
  cross-check between two genuinely independent autodiff strategies for
  the PH derivative term (``jax.grad``/``jax.hessian`` + einsum vs. a
  Cholesky-shifted ``folx.forward_laplacian`` pass).

- ``test_ph_derivative_matches_operator_definition_for_one_electron_constant_l2``:
  independent anchor for the production decomposition of the PH
  derivative operator. The reference path differentiates the diffusion
  tensor ``d`` directly with ``jax.jacfwd`` and assembles
  ``-0.5 (d:H + (div d) . g + g^T d g)`` from its operator definition,
  without reusing production's ``M = d/2 + I/2`` reformulation or
  analytic first-order vector formula.

The remaining tests pin user-visible behavior of ``PHEnergy``: residual
zero-order channel near the nucleus, init-time validation, and
backend-uniform NaN propagation under non-PD ``M``.
"""

from __future__ import annotations

from typing import Literal

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.estimator.ph import PHEnergy
from jaqmc.utils.atomic import PP_PH, core_electrons_by_pp
from jaqmc.utils.atomic.elements import from_symbol


def _gaussian_logpsi(scale: float):
    def logpsi(params, data):
        del params
        return -scale * jnp.sum(data.electrons**2)

    return logpsi


def _ph_effective_charge(symbol: str) -> int:
    return from_symbol[symbol].atomic_number - core_electrons_by_pp(symbol, PP_PH)


def _potential_energy(data: MoleculeData) -> jnp.ndarray:
    electrons = data.electrons
    atoms = data.atoms
    charges = data.charges
    r_ae = jnp.linalg.norm(electrons[:, None, :] - atoms[None, :, :], axis=-1)
    r_ee = jnp.linalg.norm(electrons[:, None, :] - electrons[None, :, :], axis=-1)
    r_aa = jnp.linalg.norm(atoms[:, None, :] - atoms[None, :, :], axis=-1)
    return (
        -jnp.sum(charges[None, :] / r_ae)
        + jnp.sum(jnp.triu(1.0 / (r_ee + jnp.eye(electrons.shape[0])), k=1))
        + jnp.sum(
            jnp.triu(
                charges[:, None] * charges[None, :] / (r_aa + jnp.eye(atoms.shape[0])),
                k=1,
            )
        )
    )


def _modern_total_ph_energy(
    *,
    data: MoleculeData,
    atom_symbols: list[str],
    ph_symbols: list[str],
    logpsi,
    kinetic_backend: Literal["standard", "forward_laplacian"] = "standard",
) -> jnp.ndarray:
    ph = PHEnergy(
        f_log_psi=logpsi,
        atom_symbols=atom_symbols,
        ph=ph_symbols,
        kinetic_backend=kinetic_backend,
    )
    ph.init(data, jax.random.key(0))
    ph_stats, _ = ph.evaluate_single_walker({}, data, {}, None, jax.random.key(1))
    return _potential_energy(data) + ph_stats["energy:kinetic"] + ph_stats["energy:ph"]


def _make_simple_ph_data(electron, ph_atom):
    return MoleculeData(
        electrons=electron[None, :],
        atoms=ph_atom[None, :],
        charges=jnp.array([_ph_effective_charge("Fe")], dtype=electron.dtype),
    )


def test_ph_local_channel_is_bounded_near_nucleus():
    """Guard the bundled PH local channel against near-nucleus blow-ups.

    The bundled paper local channel ``tilde_v_loc(r)`` must remain bounded
    when an electron sits on top of a PH nucleus. The XML loader stores
    ``loc_data(r) = r * tilde_v_loc(r) + Z_a``, so ``energy:ph`` alone
    carries a ``+Z_a / r`` Coulomb pole at the origin that cancels the
    ``-Z_a / r`` pole supplied by ``potential_energy``; only the sum
    ``energy:potential + energy:ph`` recovers the regularized
    ``tilde_v_loc(r)`` that should stay bounded for a well-formed PH table.

    This single-electron, single-atom geometry isolates the loader-side
    convention (``+ Z_a`` shift in ``data.py``) and the estimator-side
    additive split (``energy:ph + energy:potential``) from any other
    contributions, and is now the only test anchoring those two contracts.
    """
    data = MoleculeData(
        electrons=jnp.array([[0.0, 0.0, 1.0e-3]], dtype=jnp.float32),
        atoms=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        charges=jnp.array([_ph_effective_charge("S")], dtype=jnp.float32),
    )
    ph = PHEnergy(
        f_log_psi=_gaussian_logpsi(0.3),
        atom_symbols=["S"],
        ph=["S"],
    )
    ph.init(data, jax.random.key(0))

    stats, _ = ph.evaluate_single_walker({}, data, {}, None, jax.random.key(0))
    bundled_local = float(stats["energy:ph"]) + float(_potential_energy(data))

    assert abs(bundled_local) < 50.0


@pytest.mark.parametrize(
    ("atom_symbols", "ph_symbols", "atoms", "electrons"),
    [
        (
            ["Fe"],
            ["Fe"],
            jnp.array([[0.0, 0.0, 0.0]]),
            jnp.array([[0.7, 0.1, -0.2], [1.3, -0.5, 0.4]]),
        ),
        (
            ["Fe", "S"],
            ["Fe", "S"],
            jnp.array([[0.0, 0.0, 0.0], [1.6, -0.2, 0.3]]),
            jnp.array([[0.6, 0.1, -0.2], [1.9, -0.4, 0.8]]),
        ),
    ],
)
def test_ph_forward_laplacian_total_energy_matches_standard_backend(
    atom_symbols: list[str],
    ph_symbols: list[str],
    atoms: jnp.ndarray,
    electrons: jnp.ndarray,
):
    """Cross-check the two PH derivative backends against each other.

    The ``standard`` backend computes the PH derivative with
    ``jax.grad``/``jax.hessian`` and explicit einsum contractions. The
    ``forward_laplacian`` backend folds the per-electron Cholesky factor
    of ``M`` into a single ``folx.forward_laplacian`` pass. The two paths
    share the mass-matrix formula but use independent autodiff strategies,
    so disagreement here points at one of the two derivative
    decompositions (not at the shared formula).
    """
    pytest.importorskip("folx")
    scale = 0.3
    logpsi = _gaussian_logpsi(scale)

    data = MoleculeData(
        electrons=electrons,
        atoms=atoms,
        charges=jnp.array(
            [_ph_effective_charge(symbol) for symbol in atom_symbols],
            dtype=electrons.dtype,
        ),
    )
    standard_value = _modern_total_ph_energy(
        data=data,
        atom_symbols=atom_symbols,
        ph_symbols=ph_symbols,
        logpsi=logpsi,
    )
    forward_laplacian_value = _modern_total_ph_energy(
        data=data,
        atom_symbols=atom_symbols,
        ph_symbols=ph_symbols,
        logpsi=logpsi,
        kinetic_backend="forward_laplacian",
    )

    np.testing.assert_allclose(
        forward_laplacian_value,
        standard_value,
        rtol=1e-5,
        atol=1e-6,
    )


def test_ph_energy_rejects_unknown_kinetic_backend_at_init():
    """Unknown ``kinetic_backend`` raises at ``init()`` rather than at evaluate."""
    data = _make_simple_ph_data(jnp.array([0.7, 0.1, -0.2]), jnp.array([0.0, 0.0, 0.0]))
    ph = PHEnergy(
        f_log_psi=_gaussian_logpsi(0.3),
        atom_symbols=["Fe"],
        ph=["Fe"],
        kinetic_backend="definitely_not_a_backend",  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="kinetic_backend"):
        ph.init(data, jax.random.key(0))


def test_ph_energy_rejects_ph_request_with_no_matching_atom_at_init():
    """A non-empty `ph` request that matches zero atoms in `atom_symbols`.

    Raises at init().
    """
    data = MoleculeData(
        electrons=jnp.array([[0.7, 0.1, -0.2]]),
        atoms=jnp.array([[0.0, 0.0, 0.0]]),
        charges=jnp.array([1.0]),
    )
    ph = PHEnergy(
        f_log_psi=_gaussian_logpsi(0.3),
        atom_symbols=["H"],
        ph=["Fe"],
    )
    with pytest.raises(ValueError, match="ph requested element"):
        ph.init(data, jax.random.key(0))


def test_ph_derivative_matches_operator_definition_for_one_electron_constant_l2():
    r"""Independent closed-form sanity check: -0.5 div(d grad psi) / psi.

    Computes the total kinetic-like operator directly from its differential
    definition using ``jax.grad``/``jax.hessian``/``jax.jacfwd`` of an
    explicit closed-form diffusion tensor and Gaussian ``log psi``, and
    compares to ``PHEnergy.evaluate_single_walker``'s ``energy:kinetic`` on a
    hand-picked constant-:math:`\ell_2` configuration.

    The reference path does not re-state the einsum decomposition used in
    ``operator.py``; it differentiates the operator definition directly so a
    sign or factor error in the production formula would not infect the
    test.
    """
    alpha = 0.5
    l2_const = 0.07
    ph_atom = jnp.array([0.3, -0.1, 0.2])
    electron = jnp.array([0.6, 0.0, -0.4])

    def logpsi_of_x(x: jnp.ndarray) -> jnp.ndarray:
        return -alpha * jnp.sum(x**2)

    def diffusion_tensor_at(x: jnp.ndarray) -> jnp.ndarray:
        # d(x) = I + 2 * sum_a l2 * (r I - r r^T / |r|), with l2 constant.
        rel = x - ph_atom
        r = jnp.linalg.norm(rel)
        eye = jnp.eye(3, dtype=x.dtype)
        return eye + 2.0 * l2_const * (r * eye - jnp.outer(rel, rel) / r)

    grad_logpsi = jax.grad(logpsi_of_x)(electron)
    hess_logpsi = jax.hessian(logpsi_of_x)(electron)
    d = diffusion_tensor_at(electron)
    d_jac = jax.jacfwd(diffusion_tensor_at)(electron)
    # (div d)_j = partial_i d_{ij} = sum over i where input axis equals first axis.
    div_d = jnp.einsum("iji->j", d_jac)
    # -0.5 div(d grad psi) / psi = -0.5 (d:H + (div d) . g + g^T d g).
    expected = -0.5 * (
        jnp.einsum("ij,ji->", d, hess_logpsi)
        + jnp.dot(div_d, grad_logpsi)
        + jnp.einsum("i,ij,j->", grad_logpsi, d, grad_logpsi)
    )

    data = _make_simple_ph_data(electron, ph_atom)
    ph = PHEnergy(
        f_log_psi=_gaussian_logpsi(alpha),
        atom_symbols=["Fe"],
        ph=["Fe"],
        kinetic_backend="standard",
    )
    ph.init(data, jax.random.key(0))
    # Override the loaded radial tables so l2 = l2_const at every distance
    # and loc = 0. The zero-order channel is irrelevant to this test (we only
    # assert against ``energy:kinetic``), but zeroing the local table keeps
    # ``energy:ph`` deterministic for any future debug print.
    ph._l2_tables = jnp.full_like(ph._l2_tables, l2_const)
    ph._loc_tables = jnp.zeros_like(ph._loc_tables)

    stats, _ = ph.evaluate_single_walker({}, data, {}, None, jax.random.key(0))

    np.testing.assert_allclose(stats["energy:kinetic"], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kinetic_backend", ["standard", "forward_laplacian"])
def test_ph_derivative_propagates_nan_on_non_positive_definite_mass_matrix(
    kinetic_backend: Literal["standard", "forward_laplacian"],
):
    """Non-PD ``M`` surfaces as NaN uniformly across both backends.

    The two backends reach the same loud failure mode through different
    mechanisms: ``_standard`` guards ``M`` explicitly with
    ``jnp.where(eigvalsh > 0, full, NaN)`` inside
    :func:`build_second_order_matrix`, while ``_forward_laplacian`` lets
    :func:`jnp.linalg.cholesky` return NaN on non-PD input. Locking both
    paths to the same NaN behavior prevents one backend from silently
    flooring negative eigenvalues in the future while the other diverges.
    """
    if kinetic_backend == "forward_laplacian":
        pytest.importorskip("folx")

    electron = jnp.array([0.6, 0.0, -0.4])
    ph_atom = jnp.array([0.0, 0.0, 0.0])
    data = _make_simple_ph_data(electron, ph_atom)

    ph = PHEnergy(
        f_log_psi=_gaussian_logpsi(0.3),
        atom_symbols=["Fe"],
        ph=["Fe"],
        kinetic_backend=kinetic_backend,
    )
    ph.init(data, jax.random.key(0))
    # Drive l2 strongly negative so the L2 contribution to M dominates and
    # produces a non-PD assembly.
    ph._l2_tables = jnp.full_like(ph._l2_tables, -10.0)

    stats, _ = ph.evaluate_single_walker({}, data, {}, None, jax.random.key(0))

    assert bool(jnp.isnan(stats["energy:kinetic"]))
