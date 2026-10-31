# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator, mean_reduce
from jaqmc.estimator.loss_grad import loss_term
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

__all__ = ["TotalEnergy"]


@configurable_dataclass
class TotalEnergy(PerWalkerEstimator):
    """Estimator that computes total energy from component energies.

    Energy components use the ``energy:`` prefix convention (e.g.
    ``energy:kinetic``, ``energy:potential``). When ``components`` is
    ``None``, all keys starting with ``energy:`` are summed automatically.

    When the total energy is complex (e.g. from a magnetic kinetic energy
    term), ``reduce`` records the complex mean under ``output_name`` and the
    real-part mean and variance under ``{output_name}_real`` and
    ``{output_name}_real_var``.

    Args:
        loss_weight: Multiplier applied to the total energy in the VMC loss
            and gradient. Set to ``0.0`` to disable.
        output_name: Name of the output total energy field (runtime dep,
            default ``"total_energy"``).
        components: Stat keys to sum. When ``None`` (default), auto-derives
            from ``prev_walker_stats`` keys starting with ``energy:``
            (runtime dep).
    """

    loss_weight: float = 1.0
    output_name: str = runtime_dep(default="total_energy")
    components: list[str] | None = runtime_dep(default=None)

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
        total: Any = 0
        for name in keys:
            if not jnp.isscalar(energy_part := prev_walker_stats[name]):
                raise ValueError(
                    f"Expected {name} to be a scalar, but got shape "
                    f"{energy_part.shape}. If you're using a custom kinetic "
                    "energy estimator, make sure evaluate_single_walker returns a "
                    "scalar per walker."
                )
            total += energy_part
        return {
            self.output_name: total,
            **loss_term(self.output_name, total, weight=self.loss_weight),
        }, state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        key = self.output_name
        energy = walker_stats[key]
        if jnp.iscomplexobj(energy):
            return {
                **mean_reduce(walker_stats, include_variance=False),
                **mean_reduce({f"{key}_real": energy.real}),
            }
        return mean_reduce(walker_stats)
