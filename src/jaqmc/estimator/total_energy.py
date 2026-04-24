# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class TotalEnergy(PerWalkerEstimator):
    """Estimator that computes total energy from component energies.

    Energy components use the ``energy:`` prefix convention (e.g.
    ``energy:kinetic``, ``energy:potential``). When ``components`` is None,
    all keys starting with ``energy:`` are summed automatically.

    When the total energy is complex (e.g. from a magnetic kinetic energy
    term), ``reduce`` splits it into real and imaginary parts so that
    variance is computed on the real part only.

    Args:
        output_name: Name of the output total energy field.
        components: Stat keys to sum. When None (default), auto-derives
            from ``prev_walker_stats`` keys starting with ``energy:``.
    """

    output_name: str = "total_energy"
    components: list[str] | None = None

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del params, data, rngs
        keys = (
            self.components
            if self.components is not None
            else [k for k in prev_walker_stats if k.startswith("energy:")]
        )
        total = 0
        for name in keys:
            if not jnp.isscalar(energy_part := prev_walker_stats[name]):
                raise ValueError(
                    f"Expected {name} to be a scalar, but got shape "
                    f"{energy_part.shape}. If you're using a custom kinetic "
                    "energy estimator, make sure evaluate_single_walker returns a "
                    "scalar per walker."
                )
            total += energy_part
        return {self.output_name: total}, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        key = self.output_name
        energy = walker_stats[key]
        if jnp.iscomplexobj(energy):
            return {
                **mean_reduce(walker_stats, include_variance=False),
                **mean_reduce({f"{key}_real": energy.real}),
            }
        return mean_reduce(walker_stats)
