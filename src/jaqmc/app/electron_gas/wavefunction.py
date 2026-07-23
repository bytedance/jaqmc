# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Periodic neural wavefunction for the homogeneous electron gas."""

from jaqmc.app.solid.wavefunction import SolidWavefunction
from jaqmc.wavefunction.output.envelope import EnvelopeType

__all__ = ["ElectronGasWavefunction"]


class ElectronGasWavefunction(SolidWavefunction):
    """Solid wavefunction without electron-nucleus features or envelope decay."""

    envelope_type: EnvelopeType = EnvelopeType.null
