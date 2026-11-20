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
from jax.extend.core import Primitive

from ..types import ArrayOrLapTuple, LaplacianHandler


def handle_convert_element_type(
    args: tuple[ArrayOrLapTuple, ...],
    kwargs: dict[str, Any],
) -> ArrayOrLapTuple:
    return args[0].astype(kwargs["new_dtype"])


DTYPE_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    jax.lax.convert_element_type_p: handle_convert_element_type,
}
