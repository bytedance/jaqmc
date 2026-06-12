# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jax import numpy as jnp

from jaqmc.utils.config import configurable_dataclass


@configurable_dataclass
class Standard:
    r"""Standard learning rate schedule.

    .. math::
        \text{lr}(t) = \text{rate} \cdot \left(
            \frac{1}{1+t/\text{delay}}\right
        )^\text{decay}

    Args:
        rate: Initial learning rate.
        delay: Delay in steps before decay starts.
        decay: Decay rate exponent.
    """

    rate: float = 0.05
    delay: float = 2000
    decay: float = 1

    def __call__(self, t: int | jnp.ndarray) -> float | jnp.ndarray:
        return self.rate * (1.0 / (1.0 + (t / self.delay))) ** self.decay


@configurable_dataclass
class Constant:
    """Constant schedule.

    Args:
        rate: The constant rate.
    """

    rate: float = 0.05

    def __call__(self, t):
        return self.rate
