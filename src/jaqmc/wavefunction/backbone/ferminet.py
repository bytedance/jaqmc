# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from flax import linen as nn
from jax import numpy as jnp

from jaqmc.utils.array import split_nonempty_channels


class FermiLayers(nn.Module):
    """FermiNet interaction layers with single- and double-electron streams.

    Each layer updates the single-electron stream by aggregating features
    from both streams, then updates the double-electron stream independently.
    Residual connections are added when input and output dimensions match.

    Args:
        nspins: Tuple of (num_spin_up, num_spin_down) electrons.
        hidden_dims: List of (single_dim, double_dim) pairs, one per layer.
        use_last_layer: If True, also update the double stream in the final
            layer and return aggregated features. If False, skip the final
            double-stream update.
    """

    nspins: tuple[int, int]
    hidden_dims: list[tuple[int, int]]
    use_last_layer: bool = False

    @nn.compact
    def __call__(self, h_one, h_two):
        layers = self.hidden_dims

        if len(layers) == 0:
            return h_one, h_two

        # Process all but last layer - both streams updated
        for hidden_dim_single, hidden_dim_double in layers[:-1]:
            h_one = self._update_single_stream(h_one, h_two, hidden_dim_single)
            h_two = self._update_double_stream(h_two, hidden_dim_double)

        # Last layer - single stream always updated
        hidden_dim_single, hidden_dim_double = layers[-1]
        h_one = self._update_single_stream(h_one, h_two, hidden_dim_single)

        if self.use_last_layer:
            h_two = self._update_double_stream(h_two, hidden_dim_double)
            return self.aggregate_features(h_one, h_two), h_two
        else:
            return h_one, h_two

    def _update_single_stream(self, h_one, h_two, hidden_dim):
        h_in = self.aggregate_features(h_one, h_two)
        h_one_new = nn.tanh(nn.Dense(hidden_dim)(h_in))
        return self.residual(h_one, h_one_new)

    def _update_double_stream(self, h_two, hidden_dim):
        h_two_new = nn.tanh(nn.Dense(hidden_dim)(h_two))
        return self.residual(h_two, h_two_new)

    def residual(self, x, y):
        if x.shape == y.shape:
            return (x + y) / jnp.sqrt(2)
        return y

    def aggregate_features(self, h_one: jnp.ndarray, h_two: jnp.ndarray):
        """Concatenate electron features with spin-channel averages.

        For each non-empty spin channel, this computes mean single-stream
        features and mean pairwise-stream features, then concatenates those
        aggregates with the original single-electron features.

        Args:
            h_one: Single-electron features of shape
                ``(n_electrons, single_dim)``.
            h_two: Pairwise electron features of shape
                ``(n_electrons, n_electrons, double_dim)``.

        Returns:
            Aggregated single-electron features with the same leading electron
            axis as ``h_one``.
        """
        g_one = [
            jnp.mean(h_one_alpha, axis=0) * jnp.ones_like(h_one)
            for h_one_alpha in split_nonempty_channels(h_one, self.nspins)
        ]
        g_two = [
            jnp.mean(h_two_alpha, axis=0)
            for h_two_alpha in split_nonempty_channels(h_two, self.nspins)
        ]
        return jnp.concatenate([h_one, *g_one, *g_two], axis=-1)
