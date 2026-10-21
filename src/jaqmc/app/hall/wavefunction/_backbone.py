# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared Hall wavefunction backbone factory."""

from typing import Literal

from flax import linen as nn

from jaqmc.wavefunction.backbone.lapnet import LapNetBackbone
from jaqmc.wavefunction.backbone.psiformer import PsiformerBackbone


def build_hall_backbone(
    *,
    backbone: Literal["psiformer", "lapnet"],
    nspins: tuple[int, int],
    num_layers: int,
    num_heads: int,
    heads_dim: int,
) -> nn.Module:
    """Build the configured Hall backbone with fixed internal defaults.

    Returns:
        The configured Hall backbone module.

    Raises:
        ValueError: If ``backbone`` is not ``"psiformer"`` or ``"lapnet"``.
    """
    match backbone:
        case "psiformer":
            return PsiformerBackbone(
                nspins=nspins,
                num_layers=num_layers,
                num_heads=num_heads,
                heads_dim=heads_dim,
                mlp_hidden_dims=(num_heads * heads_dim,),
            )
        case "lapnet":
            return LapNetBackbone(
                nspins=nspins,
                num_layers=num_layers,
                num_heads=num_heads,
                heads_dim=heads_dim,
                use_layernorm=True,
            )
        case _:
            raise ValueError(
                f"Unsupported Hall backbone {backbone!r}. "
                "Expected 'psiformer' or 'lapnet'."
            )
