# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Determinant-state adapters for variational subspace Monte Carlo."""

import dataclasses
from dataclasses import dataclass
from typing import Protocol

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.utils.subspace_linalg import row_scaled_matrix, stable_complex_logdet
from jaqmc.wavefunction.base import WavefunctionLike


@dataclass(frozen=True)
class SubspaceSpec:
    """Describe the state/replica axis embedded in one physical walker."""

    n_states: int
    replica_fields: tuple[str, ...] = ("electrons",)

    def __post_init__(self):
        if self.n_states < 1:
            raise ValueError("n_states must be positive")
        if not self.replica_fields:
            raise ValueError("replica_fields must not be empty")


def take_replica(data: Data, replica_index: int, spec: SubspaceSpec) -> Data:
    """Remove the replica axis from selected fields of one determinant walker."""
    missing = set(spec.replica_fields) - set(data.field_names)
    if missing:
        raise KeyError(f"Replica fields are absent from data: {sorted(missing)}")
    return dataclasses.replace(
        data,
        **{name: data[name][replica_index] for name in spec.replica_fields},
    )


class IndependentStateBundle:
    """Evaluate one JaQMC ansatz architecture with independent state parameters."""

    def __init__(self, base_wavefunction: WavefunctionLike, spec: SubspaceSpec):
        self.base_wavefunction = base_wavefunction
        self.spec = spec

    def init_params(self, physical_data: Data, rngs: PRNGKey) -> Params:
        """Initialize independent parameters and stack every PyTree leaf."""
        # Preserve exact native initialization semantics for the M=1
        # regression path.  Splitting a single key would otherwise produce a
        # statistically equivalent, but different, initial wavefunction.
        keys = (
            jnp.expand_dims(rngs, 0)
            if self.spec.n_states == 1
            else jax.random.split(rngs, self.spec.n_states)
        )
        params = [
            self.base_wavefunction.init_params(physical_data, key) for key in keys
        ]
        return jax.tree.map(lambda *xs: jnp.stack(xs), *params)

    def phase_logpsi_all_states(
        self, stacked_params: Params, physical_data: Data
    ) -> tuple[jax.Array, jax.Array]:
        """Return component phases and log amplitudes with shape ``[M]``."""
        return jax.vmap(self.base_wavefunction.phase_logpsi, in_axes=(0, None))(
            stacked_params, physical_data
        )

    def complex_logpsi_all_states(
        self, stacked_params: Params, physical_data: Data
    ) -> jax.Array:
        """Return phase-aware complex log amplitudes with shape ``[M]``."""
        phase, logabs = self.phase_logpsi_all_states(stacked_params, physical_data)
        phase = phase.astype(jnp.result_type(phase, 1j))
        return logabs + 1j * jnp.angle(phase)

    def logpsi_matrix(self, stacked_params: Params, replica_data: Data) -> jax.Array:
        """Return ``L[r, s] = log(psi_s(R_r))`` with shape ``[M, M]``."""
        physical_axis = dataclasses.replace(
            replica_data,
            **{
                name: 0 if name in self.spec.replica_fields else None
                for name in replica_data.field_names
            },
        )
        return jax.vmap(
            self.complex_logpsi_all_states, in_axes=(None, physical_axis)
        )(stacked_params, replica_data)


class StateBundleLike(Protocol):
    """Internal adapter contract used by the determinant wrapper."""

    spec: SubspaceSpec

    def init_params(self, physical_data: Data, rngs: PRNGKey) -> Params: ...

    def complex_logpsi_all_states(
        self, stacked_params: Params, physical_data: Data
    ) -> jax.Array: ...

    def logpsi_matrix(
        self, stacked_params: Params, replica_data: Data
    ) -> jax.Array: ...


class DeterminantStateWavefunction:
    """Wavefunction whose amplitude is a determinant of component states."""

    def __init__(
        self,
        base_wavefunction: WavefunctionLike,
        spec: SubspaceSpec,
        *,
        state_bundle: StateBundleLike | None = None,
    ):
        self.base_wavefunction = base_wavefunction
        self.spec = spec
        self.bundle = state_bundle or IndependentStateBundle(base_wavefunction, spec)
        if self.bundle.spec != spec:
            raise ValueError("state_bundle.spec must match determinant spec")

    def init_params(self, data: Data, rngs: PRNGKey) -> Params:
        physical_data = take_replica(data, 0, self.spec)
        return self.bundle.init_params(physical_data, rngs)

    def component_logpsi_matrix(self, params: Params, data: Data) -> jax.Array:
        return self.bundle.logpsi_matrix(params, data)

    def component_amplitude_row(
        self, params: Params, data: Data
    ) -> tuple[jax.Array, jax.Array]:
        """Return a scaled component row and its removed real log scale."""
        logs = self.bundle.complex_logpsi_all_states(params, data)
        shift = jax.lax.stop_gradient(jnp.max(jnp.real(logs)))
        return jnp.exp(logs - shift), shift

    def stable_amplitude_matrix(
        self, params: Params, data: Data
    ) -> tuple[jax.Array, jax.Array]:
        return row_scaled_matrix(self.component_logpsi_matrix(params, data))

    def logpsi(self, params: Params, data: Data) -> jax.Array:
        return stable_complex_logdet(self.component_logpsi_matrix(params, data))

    def phase_logpsi(
        self, params: Params, data: Data
    ) -> tuple[jax.Array, jax.Array]:
        logpsi = self.logpsi(params, data)
        return jnp.exp(1j * jnp.imag(logpsi)), jnp.real(logpsi)

    def evaluate(self, params: Params, data: Data) -> dict[str, jax.Array]:
        return {"logpsi": self.logpsi(params, data)}
