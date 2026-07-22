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

"""Forward Laplacian rules for dtype-changing primitives."""

from typing import Any

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive

from ..guards import is_sparse_laptuple
from ..types import ArrayOrLapTuple, LaplacianHandler, LapTuple


def handle_convert_element_type(
    args: tuple[ArrayOrLapTuple, ...],
    kwargs: dict[str, Any],
) -> ArrayOrLapTuple:
    x = args[0]
    if is_sparse_laptuple(x):
        new_dtype = kwargs["new_dtype"]
        if jax.dtypes.issubdtype(new_dtype, jnp.floating) or jax.dtypes.issubdtype(
            new_dtype, jnp.complexfloating
        ):
            return LapTuple(
                x.x.astype(new_dtype),
                x.jacobian.astype(new_dtype),
                x.laplacian.astype(new_dtype),
            )
    return args[0].astype(kwargs["new_dtype"])


DTYPE_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    jax.lax.convert_element_type_p: handle_convert_element_type,
}
