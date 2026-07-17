# Copyright 2023 Microsoft Corporation
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2026.
#
# Original file was released under MIT, with the full license text
# available at licenses/folx_MIT.txt
#
# This file is distributed under the Apache License 2.0,
# with portions originally licensed under the MIT License.

"""Forward Laplacian rules for logical and predicate primitives."""

import jax
from jax.extend.core import Primitive

from ..types import LaplacianHandler
from .core import wrap_without_fwd_laplacian

LOGICAL_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    p: wrap_without_fwd_laplacian(p.bind)
    for p in [
        jax.lax.eq_p,
        jax.lax.lt_p,
        jax.lax.le_p,
        jax.lax.gt_p,
        jax.lax.ge_p,
        jax.lax.ne_p,
        jax.lax.is_finite_p,
    ]
}
for _name in ["and_p", "or_p", "xor_p", "not_p"]:
    if hasattr(jax.lax, _name):
        _primitive = getattr(jax.lax, _name)
        LOGICAL_HANDLERS[_primitive] = wrap_without_fwd_laplacian(_primitive.bind)
