# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from abc import ABCMeta
from typing import dataclass_transform

import jax


@dataclass_transform()
class JAXDataclassMeta(ABCMeta):
    """Metaclass class that turns classes into JAX-aware dataclasses.

    Classes are wrapped with :func:`dataclasses.dataclass` and registered as
    JAX pytrees. This enables instances to participate in JAX transformations
    (e.g. ``jit``, ``vmap``) while still behaving like standard dataclasses.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = dataclasses.dataclass(
            super().__new__(mcs, name, bases, namespace, **kwargs)
        )

        data_fields = [field.name for field in dataclasses.fields(cls)]

        def flatten_with_keys(x):
            return [
                (jax.tree_util.GetAttrKey(n), getattr(x, n)) for n in data_fields
            ], None

        def unflatten(_, children):
            if children and type(children[0]) is object:
                x = object.__new__(cls)
                for field, child in zip(data_fields, children):
                    setattr(x, field, child)
                return x
            return cls(**dict(zip(data_fields, children)))

        jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten)
        return cls
