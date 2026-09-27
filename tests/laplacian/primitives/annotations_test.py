# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Graph-annotation primitive semantics."""

import jax.numpy as jnp
import kfac_jax

from jaqmc.laplacian.primitives.registry import get_laplacian
from jaqmc.optimizer.kfac.tags import register_first_order_parameter

from .helpers import check_unary, parametrize_over_tracked_cases


@parametrize_over_tracked_cases("case")
def test_kfac_layer_tag_is_derivative_identity(case):
    assert get_laplacian(kfac_jax.layers_and_loss_tags.layer_tag) is not None
    check_unary(
        lambda value: register_first_order_parameter(jnp.sum(value**2), jnp.ones((1,))),
        case,
    )
