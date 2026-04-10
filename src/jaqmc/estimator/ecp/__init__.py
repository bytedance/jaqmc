# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Effective Core Potential (ECP) estimator for molecular QMC.

This module provides energy estimators for systems with effective core
potentials (pseudopotentials). ECPs replace core electrons with an
effective potential, reducing computational cost for heavy elements.

The ECP energy consists of two parts:
- Local (:math:`l=0`): A radial potential applied directly
- Nonlocal (:math:`l>0`): Angular integrals requiring wavefunction ratio evaluation

Example::

    from jaqmc.estimator.ecp import ECPEnergy
    ecp_est = ECPEnergy(
        phase_logpsi=wf.phase_logpsi,
        ecp_coefficients=pyscf_mol._ecp,
        atom_symbols=[a.symbol for a in system_config.atoms],
    )
"""

from .estimator import ECPEnergy
from .quadrature import Icosahedron, Octahedron, Quadrature, get_quadrature

__all__ = [
    "ECPEnergy",
    "Icosahedron",
    "Octahedron",
    "Quadrature",
    "get_quadrature",
]
