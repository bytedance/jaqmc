# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Potential energy estimator for the Haldane sphere."""

from collections.abc import Mapping
from typing import Any

from jax import numpy as jnp
from jax.numpy import cos, sin

from jaqmc.app.hall.config import InteractionType
from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class SpherePotential(PerWalkerEstimator):
    r"""Potential energy on the Haldane sphere.

    Converts spherical coordinates to Cartesian, computes pairwise
    cosines, then evaluates the chosen interaction.

    Args:
        interaction_type: Interaction potential form.
        monopole_strength: :math:`Q = \text{flux}/2`.
        radius: Sphere radius.
        interaction_strength: Overall scaling factor.
    """

    interaction_type: InteractionType = InteractionType.coulomb
    monopole_strength: float = 1.0
    radius: float = 1.0
    interaction_strength: float = 1.0

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del params, prev_walker_stats, rngs
        electrons = data["electrons"]
        theta, phi = electrons[..., 0], electrons[..., 1]

        # Convert to Cartesian on the unit sphere
        xyz = jnp.stack(
            [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)], axis=-1
        )
        cos12 = jnp.einsum("ia,ja->ij", xyz, xyz)

        if self.interaction_type == InteractionType.coulomb:
            r_ee = jnp.sqrt(2 - 2 * cos12)
            potential = jnp.sum(jnp.triu(1 / r_ee, k=1)) / self.radius
        else:
            raise ValueError(f"Unknown interaction type: {self.interaction_type}")

        return {"energy:potential": potential * self.interaction_strength}, state
