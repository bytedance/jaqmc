# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the three-dimensional homogeneous electron gas."""

import math

import numpy as np
import serde

from jaqmc.utils.config import configurable_dataclass

__all__ = ["ElectronGasConfig"]


def _positive_nelectrons(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"nelectrons must be a positive integer. Got {value!r}.")
    return value


@configurable_dataclass
class ElectronGasConfig:
    r"""A spin-resolved electron gas in a simple-cubic simulation cell.

    The cell volume is defined by the Wigner-Seitz radius ``rs``:

    .. math::
        \Omega = \frac{4\pi}{3} N r_s^3,

    where :math:`N=n_\uparrow+n_\downarrow`. Lengths are in Bohr.

    Args:
        rs: Wigner-Seitz radius in Bohr.
        nelectrons: Total number of electrons.
        s_z: Total spin along the z direction, ``(n_up - n_down) / 2``.
        twist: Twist in fractional reciprocal-cell coordinates. Integer shifts
            are physically equivalent.
    """

    rs: float
    nelectrons: int = serde.field(deserializer=_positive_nelectrons)
    s_z: float
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.rs) or self.rs <= 0:
            raise ValueError(f"rs must be finite and positive. Got {self.rs!r}.")
        _positive_nelectrons(self.nelectrons)
        if not math.isfinite(self.s_z) or not math.isclose(
            2 * self.s_z, round(2 * self.s_z)
        ):
            raise ValueError(f"s_z must be a finite half integer. Got {self.s_z!r}.")
        spin_imbalance = round(2 * self.s_z)
        if (
            abs(spin_imbalance) > self.nelectrons
            or (self.nelectrons + spin_imbalance) % 2 != 0
        ):
            raise ValueError(
                f"Impossible s_z={self.s_z} for {self.nelectrons} electrons."
            )
        if len(self.twist) != 3 or not np.all(np.isfinite(self.twist)):
            raise ValueError(
                "twist must contain three finite fractional coordinates. "
                f"Got {self.twist!r}."
            )

    @property
    def nspins(self) -> tuple[int, int]:
        """Return the derived ``(n_up, n_down)`` electron counts."""
        spin_imbalance = round(2 * self.s_z)
        return (
            (self.nelectrons + spin_imbalance) // 2,
            (self.nelectrons - spin_imbalance) // 2,
        )

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
