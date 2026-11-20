# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Layer normalization with local parameter rounding."""

from collections.abc import Callable
from typing import Any

from flax import linen as nn
from jax.typing import DTypeLike


class LayerNorm(nn.LayerNorm):
    """Flax LayerNorm with separately controlled parameter compute precision.

    The stored scale and bias retain ``param_dtype`` (FP32 by default). Only
    their values used by normalization are rounded, leaving Flax's FP32
    reductions and the output ``dtype`` independent of parameter rounding.
    The class name preserves Flax's automatic ``LayerNorm_N`` checkpoint paths.

    Args:
        param_compute_dtype: Dtype for scale and bias during application.
            ``None`` keeps their stored dtype.
    """

    param_compute_dtype: DTypeLike | None = None

    def param(
        self,
        name: str,
        init_fn: Callable[..., Any],
        *init_args: Any,
        unbox: bool = True,
        **init_kwargs: Any,
    ) -> Any:
        value: Any = super().param(
            name, init_fn, *init_args, unbox=unbox, **init_kwargs
        )
        if self.param_compute_dtype is not None:
            value = value.astype(self.param_compute_dtype)
        return value
