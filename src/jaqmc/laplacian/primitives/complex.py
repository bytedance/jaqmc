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
from jax import numpy as jnp
from jax.extend.core import Primitive

from ..types import LaplacianHandler, LapTuple
from .arithmetic import handle_add, handle_mul
from .core import wrap_componentwise
from .dtype import handle_convert_element_type


def handle_complex(args, kwargs):
    """Propagate the linear ``complex(real, imag)`` primitive exactly.

    ``complex(real, imag) = real + 1j * imag`` has no Hessian contribution.
    Reuse the existing sparse-aware scale and add rules so compatible sparse
    operands retain their ownership topology without taking the generic dense
    Hessian fallback.

    Returns:
        The complex-valued primal and propagated derivative state.
    """
    del kwargs
    real, imag = args
    real_value = real.x if isinstance(real, LapTuple) else real
    imag_value = imag.x if isinstance(imag, LapTuple) else imag
    complex_dtype = jnp.result_type(real_value, imag_value, 1j)
    conversion_kwargs = {"new_dtype": complex_dtype}
    real_complex = handle_convert_element_type((real,), conversion_kwargs)
    imag_complex = handle_convert_element_type((imag,), conversion_kwargs)
    scaled_imag = handle_mul(
        (imag_complex, jnp.asarray(1j, dtype=complex_dtype)),
        {},
    )
    result = handle_add((real_complex, scaled_imag), {})
    if not isinstance(result, LapTuple):
        return result

    return LapTuple(
        jax.lax.complex(real_value, imag_value),
        result.jacobian,
        result.laplacian,
    )


COMPLEX_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    p: wrap_componentwise(p.bind)
    for p in [jax.lax.conj_p, jax.lax.real_p, jax.lax.imag_p]
}
COMPLEX_HANDLERS[jax.lax.complex_p] = handle_complex
