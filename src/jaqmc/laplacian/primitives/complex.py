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

"""Forward Laplacian rules for complex component primitives."""

import jax
from jax.extend.core import Primitive

from ..types import LaplacianHandler
from .core import wrap_componentwise

COMPLEX_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    p: wrap_componentwise(p.bind)
    for p in [jax.lax.conj_p, jax.lax.real_p, jax.lax.imag_p]
}
