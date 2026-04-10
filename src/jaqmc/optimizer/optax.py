# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import field, fields, make_dataclass
from functools import cache
from typing import Any

import jax
import optax

from jaqmc.optimizer.schedule import Constant, Standard
from jaqmc.utils.config import configurable_dataclass, module_config


def __getattr__(name: str) -> Any:
    if not hasattr(optax, name):
        raise AttributeError(f"optax has no attribute {name!r}.")
    factory = getattr(optax, name)
    if not callable(factory):
        raise AttributeError(f"{name} is not a valid optimizer in optax.")
    return _make_optax_dataclass(name, factory)


@cache
def _make_optax_dataclass(name: str, factory):
    """Build a dataclass wrapper from an optax optimizer factory function.

    Args:
        name: The optax optimizer name.
        factory: The optax factory function.

    Returns:
        A dataclass type wrapping the optax optimizer.

    Raises:
        ValueError: Fail to parse optax optimizer signature to make it configurable.
    """
    fields_list: list[tuple[str, type, Any] | tuple[str, type]] = []
    for param in inspect.signature(factory).parameters.values():
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        annotation = param.annotation
        if annotation == optax.ScalarOrSchedule:
            if isinstance(param.default, (int, float)):
                mc = module_config(Constant, rate=param.default)
            else:
                mc = module_config(Standard)
            fields_list.append((param.name, Any, mc))
        elif annotation == jax.typing.ArrayLike:
            if param.default is not inspect.Parameter.empty:
                fields_list.append((param.name, float, field(default=param.default)))
            else:
                fields_list.append((param.name, float))
        elif param.default is inspect.Parameter.empty:
            # Unable to parse the type of a required parameter
            raise ValueError(
                f"Failed to make optax optimizer {name} configurable: "
                f"Unable to parse type {annotation}."
            )

    def init_method(self, params, **extra):
        del extra
        kwargs = {f.name: getattr(self, f.name) for f in fields(self)}
        self._tx = factory(**kwargs)
        return self._tx.init(params)

    def update_method(self, grads, state, params, **extra):
        del extra
        return self._tx.update(grads, state, params)

    namespace = {
        "init": init_method,
        "update": update_method,
    }

    cls = make_dataclass(name, fields_list, namespace=namespace)
    cls.__module__ = __name__
    cls.__qualname__ = name
    cls.__doc__ = inspect.getdoc(factory)
    configurable_dataclass(cls)
    return cls
