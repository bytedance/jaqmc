# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the three-dimensional homogeneous electron gas."""

import math

import numpy as np

from jaqmc.utils.config import configurable_dataclass

__all__ = ["ElectronGasConfig"]


@configurable_dataclass
class ElectronGasConfig:
    r"""A spin-resolved electron gas in a simple-cubic simulation cell.

    The cell volume is defined by the Wigner-Seitz radius ``rs``:

    .. math::
        \Omega = \frac{4\pi}{3} N r_s^3,

    where :math:`N=n_\uparrow+n_\downarrow`. Lengths are in Bohr.

    Args:
        rs: Wigner-Seitz radius in Bohr.
        nspins: Number of spin-up and spin-down electrons.
        twist: Twist in fractional reciprocal-cell coordinates. Integer shifts
            are physically equivalent.
    """

    rs: float = 1.0
    nspins: tuple[int, int] = (7, 7)
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.rs) or self.rs <= 0:
            raise ValueError(f"rs must be finite and positive. Got {self.rs!r}.")
        if len(self.nspins) != 2 or any(n < 0 for n in self.nspins):
            raise ValueError(
                "nspins must contain two non-negative electron counts. "
                f"Got {self.nspins!r}."
            )
        if self.nelectrons == 0:
            raise ValueError("At least one electron is required.")
        if len(self.twist) != 3 or not np.all(np.isfinite(self.twist)):
            raise ValueError(
                "twist must contain three finite fractional coordinates. "
                f"Got {self.twist!r}."
            )

    @property
    def nelectrons(self) -> int:
        """Total number of electrons."""
        return sum(self.nspins)

    @property
    def volume(self) -> float:
        """Simulation-cell volume in Bohr cubed."""
        return 4.0 * math.pi * self.nelectrons * self.rs**3 / 3.0

    @property
    def side_length(self) -> float:
        """Side length of the simple-cubic simulation cell in Bohr."""
        return self.volume ** (1.0 / 3.0)

    @property
    def lattice(self) -> np.ndarray:
        """Simple-cubic lattice vectors, stored row-wise."""
        return np.eye(3) * self.side_length
