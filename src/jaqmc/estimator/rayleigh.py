# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Rayleigh-matrix estimators built from JaQMC physical-energy components."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import Estimator, PerWalkerEstimator
from jaqmc.utils import parallel_jax
from jaqmc.utils.chunked_vmap import chunked_vmap
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.subspace_linalg import (
    rayleigh_solve,
    row_scaled_matrix,
    singular_value_diagnostics,
    solve_residual,
)
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.determinant_state import (
    SubspaceSpec,
    take_replica,
    take_replica_dynamic,
)


@dataclass(frozen=True)
class PhysicalEnergyPlan:
    """Classify native estimators by their replica/state dependence."""

    replica_only: tuple[str, ...] = ("potential",)

    def split(self, estimators: Mapping[str, Any]):
        replica_only = tuple(name for name in estimators if name in self.replica_only)
        pair_dependent = tuple(name for name in estimators if name not in replica_only)
        return replica_only, pair_dependent


class CrossLocalEnergyEvaluator:
    """Apply an existing JaQMC physical-energy pipeline to all ``(r, s)`` pairs."""

    def __init__(
        self,
        estimators: Mapping[str, Estimator | Any],
        spec: SubspaceSpec,
        *,
        pair_chunk_size: int | None = None,
        plan: PhysicalEnergyPlan | None = None,
    ):
        self.estimators = dict(estimators)
        self.spec = spec
        self.pair_chunk_size = pair_chunk_size
        self.plan = plan or PhysicalEnergyPlan()
        self.replica_only, self.pair_dependent = self.plan.split(self.estimators)
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

    def _run_names(
        self,
        names: tuple[str, ...],
        params: Params,
        data: Data,
        stats: dict[str, Any],
        rngs: PRNGKey,
    ) -> dict[str, Any]:
        keys = jax.random.split(rngs, len(self.estimators))
        for (name, estimator), key in zip(self.estimators.items(), keys):
            if name not in names:
                continue
            if isinstance(estimator, Estimator):
                part, _ = estimator.evaluate_single_walker(
                    params, data, stats, self._states.get(name), key
                )
            else:
                part, _ = estimator(
                    params, data, stats, self._states.get(name), key
                )
            stats.update(part)
        return stats

    def _replica_stats(
        self, stacked_params: Params, replica_data: Data, replica_index: jax.Array,
        rngs: PRNGKey,
    ) -> dict[str, Any]:
        params = jax.tree.map(
            lambda x: jax.lax.dynamic_index_in_dim(x, 0, axis=0, keepdims=False),
            stacked_params,
        )
        data = take_replica_dynamic(replica_data, replica_index, self.spec)
        return self._run_names(self.replica_only, params, data, {}, rngs)

    def _pair_local_energy(
        self,
        stacked_params: Params,
        replica_data: Data,
        replica_stats: Mapping[str, Any],
        pair_index: jax.Array,
        rngs: PRNGKey,
    ) -> jax.Array:
        m = self.spec.n_states
        replica_index, state_index = pair_index // m, pair_index % m
        params = jax.tree.map(
            lambda x: jax.lax.dynamic_index_in_dim(
                x, state_index, axis=0, keepdims=False
            ),
            stacked_params,
        )
        data = take_replica_dynamic(replica_data, replica_index, self.spec)
        stats = jax.tree.map(
            lambda x: jax.lax.dynamic_index_in_dim(
                x, replica_index, axis=0, keepdims=False
            ),
            dict(replica_stats),
        )
        stats = self._run_names(self.pair_dependent, params, data, stats, rngs)
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
        replica_rngs, pair_rngs = jax.random.split(rngs)
        replica_stats = jax.vmap(
            self._replica_stats, in_axes=(None, None, 0, 0)
        )(
            stacked_params,
            replica_data,
            jnp.arange(m),
            jax.random.split(replica_rngs, m),
        )
        pair_indices = jnp.arange(m * m)
        keys = jax.random.split(pair_rngs, m * m)
        values = chunked_vmap(
            self._pair_local_energy,
            in_axes=(None, None, None, 0, 0),
            chunk_size=self.pair_chunk_size,
        )(stacked_params, replica_data, replica_stats, pair_indices, keys)
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
        phi_finite = jnp.all(jnp.isfinite(phi))
        # Derive the sentinel from ``phi`` so it inherits the same varying-axis
        # annotation inside shard_map as the SVD branch outputs.
        nan_real = jnp.real(phi[0, 0]) * 0 + jnp.nan
        sigma_min, sigma_max, condition = jax.lax.cond(
            phi_finite,
            lambda _: singular_value_diagnostics(phi),
            lambda _: (nan_real, nan_real, nan_real),
            operand=None,
        )
        matrix_finite = jnp.all(jnp.isfinite(phi)) & jnp.all(jnp.isfinite(phi_h))
        rayleigh = jax.lax.cond(
            matrix_finite,
            lambda _: rayleigh_solve(phi, phi_h),
            lambda _: jnp.full_like(phi_h, jnp.nan),
            operand=None,
        )
        residual = solve_residual(phi, rayleigh, phi_h)
        rayleigh_valid = (
            matrix_finite
            & jnp.all(jnp.isfinite(rayleigh))
            & jnp.isfinite(residual)
        )
        trace = jnp.trace(rayleigh)
        return {
            "local_rayleigh": rayleigh,
            "subspace_local_energy": trace,
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
            "rayleigh_finite": rayleigh_valid,
            "rayleigh_valid": rayleigh_valid,
        }, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        local_rayleigh = walker_stats["local_rayleigh"]
        valid = walker_stats["rayleigh_valid"] & jnp.all(
            jnp.isfinite(local_rayleigh), axis=(-2, -1)
        )
        local_count = jnp.sum(valid)
        global_count = parallel_jax.psum(local_count)
        global_total = parallel_jax.psum(jnp.asarray(valid.shape[0]))
        rayleigh_mean = (
            parallel_jax.psum(jnp.sum(local_rayleigh, axis=0)) / global_total
        )
        second_moment = (
            parallel_jax.psum(jnp.sum(jnp.abs(local_rayleigh) ** 2, axis=0))
            / global_total
        )
        rayleigh_var = jnp.maximum(second_moment - jnp.abs(rayleigh_mean) ** 2, 0)
        valid_fraction = global_count / global_total
        invalid_count = global_total - global_count

        reduced = {}
        for key, value in walker_stats.items():
            if key == "local_rayleigh":
                continue
            reduced[key] = (
                parallel_jax.psum(jnp.sum(value, axis=0)) / global_total
            )
        subspace_energy = walker_stats["subspace_energy"]
        energy_second_moment = (
            parallel_jax.psum(jnp.sum(subspace_energy**2, axis=0)) / global_total
        )
        reduced["subspace_energy_var"] = jnp.maximum(
            energy_second_moment - reduced["subspace_energy"] ** 2, 0
        )
        step_valid = global_count == global_total
        eig_input = jnp.where(step_valid, rayleigh_mean, jnp.zeros_like(rayleigh_mean))
        eigenvalues, eigenvectors = jnp.linalg.eig(eig_input)
        order = jnp.argsort(jnp.real(eigenvalues))
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        max_ritz_imag = jnp.max(jnp.abs(jnp.imag(eigenvalues)))
        return {
            **reduced,
            "rayleigh_mean": rayleigh_mean,
            "local_rayleigh_variance": rayleigh_var,
            "rayleigh_valid_fraction": valid_fraction,
            "rayleigh_invalid_count": invalid_count,
            "training_step_valid": step_valid,
            "ritz_energies": jnp.real(eigenvalues),
            "ritz_energies_imag": jnp.imag(eigenvalues),
            "ritz_vectors": eigenvectors,
            "max_ritz_imag": max_ritz_imag,
            "max_ritz_imag_warning": (
                max_ritz_imag > self.max_imag_eigenvalue_warning
            ),
        }
