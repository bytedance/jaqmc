# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the quantum Hall effect on a Haldane sphere."""

from enum import StrEnum

from jaqmc.utils.config import configurable_dataclass


class InteractionType(StrEnum):
    r"""Electron-electron interaction type on the Haldane sphere.

    Attributes:
        coulomb: Coulomb repulsion :math:`1/r_{ij}`.
    """

    coulomb = "coulomb"


@configurable_dataclass
class HallConfig:
    r"""Configuration for a quantum Hall system on the Haldane sphere.

    Args:
        flux: Magnetic flux :math:`2Q` (positive integer).
        nspins: ``(n_up, n_down)`` electron counts.
        radius: Sphere radius. Defaults to :math:`\sqrt{Q}`.
        interaction_type: Interaction potential form.
        interaction_strength: Scaling factor for the potential energy.
        lz_center: Target :math:`L_z` for the penalty method.
        lz_penalty: Penalty strength for
            :math:`(L_z - L_{z,0})^2`.
        l2_penalty: Penalty strength for :math:`L^2`.
    """

    flux: int = 2
    nspins: tuple[int, int] = (3, 0)
    radius: float | None = None
    interaction_type: InteractionType = InteractionType.coulomb
    interaction_strength: float = 1.0
    lz_center: float = 0.0
    lz_penalty: float = 0.0
    l2_penalty: float = 0.0
