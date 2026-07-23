# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Periodic neural wavefunction for the homogeneous electron gas."""

import jax
from jax import numpy as jnp

from jaqmc.app.solid.data import SolidData
from jaqmc.app.solid.wavefunction import SolidWavefunction
from jaqmc.array_types import Params, PRNGKey
from jaqmc.wavefunction.output.envelope import EnvelopeType

__all__ = ["ElectronGasWavefunction"]


class ElectronGasWavefunction(SolidWavefunction):
    """Solid wavefunction without electron-nucleus features or envelope decay."""

    envelope_type: EnvelopeType = EnvelopeType.null
    parameter_dtype: str = "float32"

    def init_params(self, data: SolidData, rngs: PRNGKey) -> Params:
        """Initialize and cast neural-network parameters to the requested dtype.

        Returns:
            Wavefunction parameters with floating leaves in ``parameter_dtype``.

        Raises:
            ValueError: If the dtype is unsupported or FP64 is disabled in JAX.
        """
        if self.parameter_dtype not in {"float32", "float64"}:
            raise ValueError(
                "parameter_dtype must be 'float32' or 'float64'. "
                f"Got {self.parameter_dtype!r}."
            )
        if self.parameter_dtype == "float64" and not jax.config.read("jax_enable_x64"):
            raise ValueError("parameter_dtype='float64' requires jax.enable_x64=true.")

        params = super().init_params(data, rngs)
        real_dtype = jnp.dtype(self.parameter_dtype)
        complex_dtype = jnp.dtype(
            "complex128" if self.parameter_dtype == "float64" else "complex64"
        )

        def cast(value):
            if jnp.issubdtype(value.dtype, jnp.floating):
                return value.astype(real_dtype)
            if jnp.issubdtype(value.dtype, jnp.complexfloating):
                return value.astype(complex_dtype)
            return value

        return jax.tree.map(cast, params)
