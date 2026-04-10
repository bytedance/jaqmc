# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from flax import linen as nn
from jax import numpy as jnp

from jaqmc.utils.array import split_nonempty_channels


class SplitChannelDense(nn.Module):
    """Apply separate dense layers to each spin channel.

    Args:
        channels: Tuple of (num_spin_up, num_spin_down) electrons.
        features: Output feature dimensions for DenseGeneral.
        use_bias: Whether to use bias in dense layers.
    """

    channels: tuple[int, int]
    features: list[int]
    use_bias: bool = True

    @nn.compact
    def __call__(self, h_one: jnp.ndarray):
        return jnp.concatenate(
            [
                nn.DenseGeneral(self.features, use_bias=self.use_bias)(h)
                for h in split_nonempty_channels(h_one, self.channels)
            ]
        )


class OrbitalProjection(nn.Module):
    """Project backbone features to orbital matrix for molecules.

    Handles both spin-split and non-split configurations:

    - Spin-split: Separate dense layers for each spin channel, allowing
      different orbital transformations for spin-up and spin-down electrons.
    - Non-split: Single dense layer shared across all electrons.

    The output is reshaped and transposed to produce the standard orbital
    matrix format ``(ndets, n_electrons, n_electrons)`` used by determinant
    layers.

    Args:
        nspins: Tuple of (num_spin_up, num_spin_down) electrons.
        ndets: Number of determinants.
        orbitals_spin_split: If True, use separate projection for each spin
            channel. Only effective when both spin channels are occupied.
        use_bias: Whether to use bias in dense layers.
    """

    nspins: tuple[int, int]
    ndets: int
    orbitals_spin_split: bool = True
    use_bias: bool = False

    @nn.compact
    def __call__(self, h_one: jnp.ndarray) -> jnp.ndarray:
        """Project single-electron features to orbital matrix.

        Args:
            h_one: Single-electron features of shape ``(n_electrons, hidden)``.

        Returns:
            Orbital matrix of shape ``(ndets, n_electrons, n_electrons)``.
        """
        n_electrons = sum(self.nspins)
        features = [self.ndets, n_electrons]
        active_spins = [s for s in self.nspins if s > 0]

        if self.orbitals_spin_split and len(active_spins) > 1:
            orbitals = SplitChannelDense(self.nspins, features, self.use_bias)(h_one)
        else:
            orbitals = nn.DenseGeneral(features, use_bias=self.use_bias)(h_one)

        return jnp.transpose(orbitals, (1, 0, 2))
