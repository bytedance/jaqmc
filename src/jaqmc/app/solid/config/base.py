# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Any

import numpy as np

from jaqmc.utils.atomic.atomic_system import AtomicSystemConfig
from jaqmc.utils.config import configurable_dataclass

__all__ = ["SolidConfig"]


@configurable_dataclass
class SolidConfig(AtomicSystemConfig):
    """Configuration for solid-state/periodic systems.

    This class holds the configuration for solid-state systems, including
    lattice vectors, supercell matrix, and twist vectors. It inherits from
    :class:`~jaqmc.utils.atomic.atomic_system.AtomicSystemConfig`.

    Attributes:
        lattice_vectors: The primitive lattice vectors as a 3x3 matrix.
        supercell_matrix: An optional 3x3 integer matrix to define a supercell.
        twist: The twist vector for boundary conditions (k-point).
        scale: An integer scaling factor representing the ratio of the supercell
            volume to the primitive cell volume.
            Note: If `supercell_matrix` is provided, this attribute is automatically
            updated in `__post_init__` to match the determinant of that matrix.
        supercell_lattice: The calculated 3x3 supercell lattice vectors.
            This is a derived attribute (init=False) populated during initialization.
    """

    lattice_vectors: list[list[float]] = field(default_factory=list)
    supercell_matrix: list[list[int]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    twist: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: int = field(init=False, metadata={"serde_skip": True})
    supercell_lattice: Any = field(init=False, metadata={"serde_skip": True})

    def __post_init__(self):
        super().__post_init__()
        lattice_arr = np.array(self.lattice_vectors)
        if lattice_arr.shape != (3, 3):
            raise ValueError(
                f"lattice_vectors must be a 3x3 matrix. Got shape {lattice_arr.shape}."
            )
        if self.supercell_matrix is not None:
            supercell_arr = np.array(self.supercell_matrix)
            if supercell_arr.shape != (3, 3):
                raise ValueError(
                    f"supercell_matrix must be a 3x3 matrix. "
                    f"Got shape {supercell_arr.shape}."
                )
            if not np.all(supercell_arr == np.round(supercell_arr)):
                raise ValueError("supercell_matrix must have integer entries.")
            self.supercell_lattice = np.dot(supercell_arr, lattice_arr).tolist()
            self.scale = round(np.linalg.det(supercell_arr))
        else:
            self.supercell_lattice = lattice_arr
