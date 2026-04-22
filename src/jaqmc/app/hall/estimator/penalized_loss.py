# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Angular momentum penalty estimator for state selection.

Adds :math:`\lambda_{L_z}(L_z - L_{z,0})^2 + \lambda_{L^2} L^2` to the
total energy so the VMC optimizer drives the wavefunction toward a target
angular momentum sector.
"""

from collections.abc import Mapping
from typing import Any

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import LocalEstimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class PenalizedLoss(LocalEstimator):
    """Adds angular momentum penalties to total energy for state selection.

    Reads ``total_energy``, ``angular_momentum_z``,
    ``angular_momentum_z_square``, and ``angular_momentum_square`` from
    ``prev_local_stats`` and outputs a ``penalized_loss`` key.

    Args:
        lz_center: Target :math:`L_z` value.
        lz_penalty: Penalty strength for :math:`(L_z - L_{z,0})^2`.
        l2_penalty: Penalty strength for :math:`L^2`.
    """

    lz_center: float = 0.0
    lz_penalty: float = 0.0
    l2_penalty: float = 0.0

    def evaluate_local(
        self,
        params: Params,
        data: Data,
        prev_local_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del params, data, rngs

        loss = prev_local_stats["total_energy"]

        if self.lz_penalty:
            lz = prev_local_stats["angular_momentum_z"]
            lz_sq = prev_local_stats["angular_momentum_z_square"]
            # (Lz - lz_center)^2 = Lz^2 - 2*lz_center*Lz + lz_center^2
            loss = loss + self.lz_penalty * (
                lz_sq - 2 * self.lz_center * lz + self.lz_center**2
            )

        if self.l2_penalty:
            l2 = prev_local_stats["angular_momentum_square"]
            loss = loss + self.l2_penalty * l2

        return {"penalized_loss": loss}, state
