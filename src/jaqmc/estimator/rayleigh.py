# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Rayleigh-matrix estimators built from JaQMC physical-energy components."""

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import Estimator, PerWalkerEstimator, mean_reduce
from jaqmc.utils import parallel_jax
from jaqmc.utils.chunked_vmap import chunked_vmap
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.subspace_linalg import (
    complex_variance,
    rayleigh_solve,
    row_scaled_matrix,
    singular_value_diagnostics,
    solve_residual,
)
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.determinant_state import SubspaceSpec, take_replica


class CrossLocalEnergyEvaluator:
    """Apply an existing JaQMC physical-energy pipeline to all ``(r, s)`` pairs."""

    def __init__(
        self,
        estimators: Mapping[str, Estimator | Any],
        spec: SubspaceSpec,
        *,
        pair_chunk_size: int | None = None,
    ):
        self.estimators = dict(estimators)
        self.spec = spec
        self.pair_chunk_size = pair_chunk_size
        self._states: dict[str, Any] = {}

    def init(self, replica_data: Data, rngs: PRNGKey) -> None:
        """Initialize reused physical estimators from one physical replica."""
        physical_data = take_replica(replica_data, 0, self.spec)
        keys = jax.random.split(rngs, len(self.estimators))
        self._states = {
            name: estimator.init(physical_data, key)
            if isinstance(estimator, Estimator)
            else None
            for (name, estimator), key in zip(self.estimators.items(), keys)
        }
        stateful = [name for name, state in self._states.items() if state is not None]
        if stateful:
            raise ValueError(
                "CrossLocalEnergyEvaluator currently requires stateless physical "
                f"estimators; got state from {stateful}"
            )

    def _single(
        self, params: Params, data: Data, rngs: PRNGKey
    ) -> jax.Array:
        stats: dict[str, Any] = {}
        keys = jax.random.split(rngs, len(self.estimators))
        for (name, estimator), key in zip(self.estimators.items(), keys):
            if isinstance(estimator, Estimator):
                part, _ = estimator.evaluate_single_walker(
                    params, data, stats, self._states.get(name), key
                )
            else:
                part, _ = estimator(
                    params, data, stats, self._states.get(name), key
                )
            stats.update(part)
        if "total_energy" in stats:
            return stats["total_energy"]
        components = [
            value for key, value in stats.items() if key.startswith("energy:")
        ]
        if not components:
            raise ValueError("Physical estimator pipeline produced no energy values")
        return sum(components[1:], start=components[0])

    def __call__(
        self, stacked_params: Params, replica_data: Data, rngs: PRNGKey
    ) -> jax.Array:
        """Return ``E_local[r, s]`` using flattened, optionally chunked pairs."""
        m = self.spec.n_states
        replica_indices = jnp.repeat(jnp.arange(m), m)
        state_indices = jnp.tile(jnp.arange(m), m)
        pair_params = jax.tree.map(lambda x: x[state_indices], stacked_params)
        pair_data = dataclasses.replace(
            replica_data,
            **{
                name: replica_data[name][replica_indices]
                for name in self.spec.replica_fields
            },
        )
        pair_axes = dataclasses.replace(
            pair_data,
            **{
                name: 0 if name in self.spec.replica_fields else None
                for name in pair_data.field_names
            },
        )
        keys = jax.random.split(rngs, m * m)
        values = chunked_vmap(
            self._single,
            in_axes=(0, pair_axes, 0),
            chunk_size=self.pair_chunk_size,
        )(pair_params, pair_data, keys)
        return values.reshape(m, m)


@configurable_dataclass
class RayleighMatrixEstimator(PerWalkerEstimator):
    """Estimate the determinant-state local Rayleigh matrix per walker."""

    matrix_dtype: str = "complex128"
    pair_chunk_size: int | None = None
    condition_warning: float = 1e10
    solve_residual_warning: float = 1e-6
    max_imag_eigenvalue_warning: float = 1e-6
    f_component_logpsi_matrix: Any = runtime_dep()
    f_cross_local_energy: Any = runtime_dep()

    def init(self, data: Data, rngs: PRNGKey) -> None:
        if hasattr(self.f_cross_local_energy, "init"):
            self.f_cross_local_energy.init(data, rngs)
        return None

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_walker_stats
        logs = self.f_component_logpsi_matrix(params, data)
        phi, _ = row_scaled_matrix(logs)
        dtype = jnp.dtype(self.matrix_dtype)
        phi = phi.astype(dtype)
        local_energy = self.f_cross_local_energy(params, data, rngs).astype(dtype)
        phi_h = phi * local_energy
        rayleigh = rayleigh_solve(phi, phi_h)
        residual = solve_residual(phi, rayleigh, phi_h)
        sigma_min, sigma_max, condition = singular_value_diagnostics(phi)
        trace = jnp.trace(rayleigh)
        return {
            "local_rayleigh": rayleigh,
            "subspace_energy": jnp.real(trace),
            "subspace_energy_imag": jnp.imag(trace),
            "rayleigh_solve_residual": residual,
            "rayleigh_solve_residual_warning": (
                ~jnp.isfinite(residual) | (residual > self.solve_residual_warning)
            ),
            "amplitude_sigma_min": sigma_min,
            "amplitude_sigma_max": sigma_max,
            "amplitude_condition": condition,
            "amplitude_condition_warning": (
                ~jnp.isfinite(condition) | (condition > self.condition_warning)
            ),
            "rayleigh_finite": jnp.all(jnp.isfinite(rayleigh)),
        }, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        local_rayleigh = walker_stats["local_rayleigh"]
        rayleigh_mean = parallel_jax.pmean(jnp.nanmean(local_rayleigh, axis=0))
        rayleigh_var = parallel_jax.pmean(complex_variance(local_rayleigh, axis=0))
        scalar_stats = {
            key: value
            for key, value in walker_stats.items()
            if key != "local_rayleigh"
        }
        reduced = mean_reduce(scalar_stats)
        eigenvalues, eigenvectors = jnp.linalg.eig(rayleigh_mean)
        order = jnp.argsort(jnp.real(eigenvalues))
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        max_ritz_imag = jnp.max(jnp.abs(jnp.imag(eigenvalues)))
        return {
            **reduced,
            "rayleigh_mean": rayleigh_mean,
            "local_rayleigh_variance": rayleigh_var,
            "ritz_energies": jnp.real(eigenvalues),
            "ritz_energies_imag": jnp.imag(eigenvalues),
            "ritz_vectors": eigenvectors,
            "max_ritz_imag": max_ritz_imag,
            "max_ritz_imag_warning": (
                max_ritz_imag > self.max_imag_eigenvalue_warning
            ),
        }
