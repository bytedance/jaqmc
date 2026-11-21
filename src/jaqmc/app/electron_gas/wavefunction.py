# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Periodic neural wavefunction for the homogeneous electron gas."""

from typing import Protocol, runtime_checkable

from jaqmc.app.solid.wavefunction import SolidWavefunction, SolidWavefunctionLike
from jaqmc.wavefunction.output.envelope import EnvelopeType

__all__ = ["ElectronGasWavefunction", "ElectronGasWavefunctionLike"]


@runtime_checkable
class ElectronGasWavefunctionLike(SolidWavefunctionLike, Protocol):
    """Runtime interface required by electron-gas workflows."""

    envelope_type: EnvelopeType


class ElectronGasWavefunction(SolidWavefunction):
    """Solid wavefunction without electron-nucleus features or envelope decay."""

    envelope_type: EnvelopeType = EnvelopeType.null
