# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Protocol, TypedDict

import serde
from flax import linen as nn
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data


class WavefunctionInit[DataT: Data](Protocol):
    def __call__(self, data: DataT, rngs: PRNGKey) -> Params:
        """Initialize wavefunction parameters from one walker sample.

        Returns:
            A PyTree representing the initial wavefunction parameters.
        """


class WavefunctionEvaluate[DataT: Data, OutputT](Protocol):
    def __call__(self, params: Params, data: DataT) -> OutputT:
        """Evaluate a wavefunction with explicit parameters.

        Args:
            params: Parameter PyTree to evaluate.
            data: One-walker input sample.

        Returns:
            Model output for this walker (scalar or structured output).
        """


type NumericWavefunctionEvaluate[DataT: Data] = WavefunctionEvaluate[DataT, jnp.ndarray]
"""Callable protocol for one-walker numeric wavefunction evaluation.

This alias specializes :class:`WavefunctionEvaluate` to scalar-array outputs,
which are typically log-amplitude values such as ``log|psi|``.
"""


class WavefunctionLike[DataT: Data, OutputT](Protocol):
    """Minimal wavefunction interface required by framework components.

    App-level protocols can extend this with domain-specific methods such as
    ``logpsi``, ``phase_logpsi``, or ``orbitals``.
    """

    init_params: WavefunctionInit[DataT]
    evaluate: WavefunctionEvaluate[DataT, OutputT]


class Wavefunction[DataT: Data, OutputT](
    nn.Module, WavefunctionLike[DataT, OutputT], ABC
):
    """Base class for JaQMC wavefunctions.

    A ``Wavefunction`` is a Flax ``nn.Module`` with three complementary
    execution interfaces:

    - ``__call__(data)``: model definition for one walker (implemented by subclasses).
    - ``apply(params, data, ...)``: Flax runtime API that executes methods with explicit
      variables.
    - ``evaluate(params, data)``: JaQMC wrapper over ``apply`` that provides a stable
      typed contract for framework consumers.
    """

    def init_params(self, data: DataT, rngs: PRNGKey) -> Params:
        """Initialize parameters from one sample walker.

        JaQMC wrapper over :meth:`~flax.linen.Module.init` that provides a stable typed
        contract, following the signature of :class:`WavefunctionInit`.

        Returns:
            A PyTree containing the initialized wavefunction parameters.
        """
        return self.init(rngs, data)

    @abstractmethod
    def __call__(self, data: DataT) -> OutputT:
        """Define the forward pass for one walker.

        This method should be written in standard Flax style and does not take
        explicit parameters.
        """

    def evaluate(self, params: Params, data: DataT) -> OutputT:
        """Framework-level execution entrypoint.

        JaQMC wrapper over :meth:`~flax.linen.Module.apply` that provides a stable typed
        contract, following the signature of :class:`WavefunctionEvaluate`.

        Returns:
            The wavefunction output for the provided parameters and data.
        """
        return self.apply(params, data)  # type: ignore

    @classmethod
    def __init_subclass__(cls):
        """Auto-register subclass for config serialization.

        Wavefunction subclasses do not need ``@configurable_dataclass``:
        this hook marks Flax internal fields as ``serde_skip`` and applies
        ``serde.serde(..., type_check=serde.coerce)``.
        """
        super().__init_subclass__(cls)
        cls.__dataclass_fields__["parent"].metadata = {"serde_skip": True}
        cls.__dataclass_fields__["name"].metadata = {"serde_skip": True}
        serde.serde(cls, type_check=serde.coerce)


class LogPsiWFOutput(TypedDict):
    """Minimal structured wavefunction output containing ``logpsi``."""

    logpsi: jnp.ndarray
    "logpsi: Log wavefunction value or log amplitude for one walker."


class ComplexWFOutput(LogPsiWFOutput, TypedDict):
    """Structured output for complex-valued wavefunctions."""


class RealWFOutput(LogPsiWFOutput, TypedDict):
    """Structured output for real-valued wavefunctions."""

    sign_logpsi: jnp.ndarray
    "Sign of the wavefunction value for one walker."
