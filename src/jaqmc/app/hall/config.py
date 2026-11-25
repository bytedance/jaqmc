# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the quantum Hall effect on a Haldane sphere."""

from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class HallConfig:
    r"""Configuration for a quantum Hall system on the Haldane sphere.

    Args:
        flux: Magnetic flux :math:`2Q` (positive integer).
        nspins: ``(n_up, n_down)`` electron counts.
        radius: Sphere radius. Defaults to :math:`\sqrt{Q}`.
        interaction_strength: Scaling factor for the Coulomb interaction.
    """

    flux: int = 2
    nspins: tuple[int, int] = (3, 0)
    radius: float | None = None
    interaction_strength: float = 1.0
