# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Samplers updates (parts of) the data according to a distribution or data pool.

Samplers should not care about the exact signature of the wavefunction. They just
take in a function and corresponding data to be sampled and updates the data.

Although some of the samplers (like those drawing from a data pool) can obtain an
initial data directly, most of other samplers (like those drawing from a distribution)
cannot. Therefore, the logic of creating an initial set of data is separated from the
main samplers logic.

Since different samplers are decoupled and only handles its own fragment of data,
they can be chained together to form a new sampler. However, to make things simpler,
the chaining logics are handled in `DataGenerator`s instead of the samplers themselves.
"""

from .base import SamplePlan, SamplerInit, SamplerLike, SamplerStep

__all__ = ["SamplePlan", "SamplerInit", "SamplerLike", "SamplerStep"]
