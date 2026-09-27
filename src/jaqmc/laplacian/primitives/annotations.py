# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Forward Laplacian rules for graph-annotation primitives."""

import kfac_jax
from jax.extend.core import Primitive

from ..types import LaplacianHandler


def _layer_tag(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Return the tagged output without changing its derivative payload."""
    (output,) = kfac_jax.layers_and_loss_tags.layer_tag.layer_data(args, kwargs).outputs
    return output


ANNOTATION_HANDLERS: dict[Primitive | str, LaplacianHandler] = {
    kfac_jax.layers_and_loss_tags.layer_tag: _layer_tag,
}
