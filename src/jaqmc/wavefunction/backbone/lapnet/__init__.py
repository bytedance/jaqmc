# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""LapNet dual-stream cross-attention backbone."""

from ._attention import lapnet_sparse_attention
from ._backbone import LapNetBackbone, LapNetLayer, QKProjection, QKStreams

__all__ = [
    "LapNetBackbone",
    "LapNetLayer",
    "QKProjection",
    "QKStreams",
    "lapnet_sparse_attention",
]
