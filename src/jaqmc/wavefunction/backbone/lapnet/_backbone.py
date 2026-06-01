# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""LapNet backbone modules."""

import jax
from flax import linen as nn
from jax import numpy as jnp

from ._attention import lapnet_sparse_attention

QKProjection = tuple[jnp.ndarray, jnp.ndarray]
QKStreams = tuple[QKProjection, ...]


class LapNetLayer(nn.Module):
    """One LapNet cross-attention block.

    This preserves the original LapNet semantics:
    `hs` supplies query/key features, `hd` supplies value features, the
    attention/MLP residuals update only `hd`, and the `hs` path advances only
    through local residual `tanh(Dense(...))` updates.

    Args:
        num_heads: Number of cross-attention heads.
        heads_dim: Dimension of each attention head.
        use_layernorm: Whether to apply LayerNorm around the attention and MLP
            sub-blocks.
        use_bias: Whether to use bias in dense projections.
        num_local_updates: Number of independent `hs` residual update layers
            between attention blocks.
    """

    num_heads: int
    heads_dim: int
    use_layernorm: bool = False
    use_bias: bool = True
    num_local_updates: int = 2

    def setup(self) -> None:
        if self.num_local_updates < 0:
            raise ValueError("LapNet num_local_updates must be non-negative.")
        attention_dim = self.num_heads * self.heads_dim
        bias_init = nn.initializers.normal(stddev=1.0)
        self.qk_projection = nn.Dense(
            2 * attention_dim, use_bias=self.use_bias, bias_init=bias_init
        )
        self.value_projection = nn.Dense(
            attention_dim, use_bias=self.use_bias, bias_init=bias_init
        )
        self.output_projection = nn.Dense(
            attention_dim, use_bias=self.use_bias, bias_init=bias_init
        )
        self.value_update = nn.Dense(
            attention_dim, use_bias=self.use_bias, bias_init=bias_init
        )
        self.qk_update_layers = [
            nn.Dense(attention_dim, use_bias=self.use_bias, bias_init=bias_init)
            for _ in range(self.num_local_updates)
        ]
        if self.use_layernorm:
            self.qk_layernorm = nn.LayerNorm(epsilon=1e-6)
            self.value_layernorm = nn.LayerNorm(epsilon=1e-6)
            self.post_attention_layernorm = nn.LayerNorm(epsilon=1e-6)

    def dense_block(
        self, qk_projection: QKProjection, value_stream: jnp.ndarray
    ) -> jnp.ndarray:
        """Update the dense stream with one attention and MLP block.

        Args:
            qk_projection: Query/key projections from the individual stream.
            value_stream: Dense stream used for value projection and residual
                updates.

        Returns:
            Updated dense stream.

        Raises:
            ValueError: If the stream widths do not match
                `num_heads * heads_dim`.
        """
        attention_dim = self.num_heads * self.heads_dim
        query, key = qk_projection
        if query.shape[-1] != attention_dim or key.shape[-1] != attention_dim:
            raise ValueError("LapNet query/key widths must match attention_dim.")
        if value_stream.shape[-1] != attention_dim:
            raise ValueError(
                "LapNet query/key and value streams must have matching feature widths."
            )

        n_electrons = query.shape[0]
        value = (
            self.value_layernorm(value_stream) if self.use_layernorm else value_stream
        )
        value = self.value_projection(value)

        query = query.reshape(n_electrons, self.num_heads, self.heads_dim)
        key = key.reshape(n_electrons, self.num_heads, self.heads_dim)
        value = value.reshape(n_electrons, self.num_heads, self.heads_dim)
        attended_values = lapnet_sparse_attention(query, key, value)
        attended_values = attended_values.reshape(n_electrons, attention_dim)
        attended_values = self.output_projection(attended_values)

        residual_value = value_stream + attended_values
        residual_norm = (
            self.post_attention_layernorm(residual_value)
            if self.use_layernorm
            else residual_value
        )
        return residual_value + nn.tanh(self.value_update(residual_norm))

    def project_qk_stream(self, qk_stream: jnp.ndarray) -> QKProjection:
        """Project one individual-stream vector into query and key features.

        Args:
            qk_stream: One-electron individual-stream feature vector.

        Returns:
            Query and key feature vectors.
        """
        qk_stream = self.qk_layernorm(qk_stream) if self.use_layernorm else qk_stream
        query, key = jnp.split(self.qk_projection(qk_stream), 2, axis=-1)
        return query, key

    def qk_stream_block(self, qk_stream: jnp.ndarray) -> jnp.ndarray:
        """Advance one individual-stream vector to the next layer.

        Args:
            qk_stream: One-electron individual-stream feature vector.

        Returns:
            Updated individual-stream feature vector.
        """
        for dense in self.qk_update_layers:
            qk_stream = qk_stream + nn.tanh(dense(qk_stream))
        return qk_stream


class LapNetBackbone(nn.Module):
    """Dual-stream LapNet backbone for molecular wavefunctions.

    The JaQMC implementation preserves the original LapNet block graph: the
    per-electron individual stream `hs` produces query/key features and advances
    only through local residual updates, while the dense stream `hd` receives
    the cross-attention and post-attention MLP residuals.

    Args:
        nspins: Tuple of spin-up and spin-down electron counts.
        num_layers: Number of LapNet layers.
        num_heads: Number of attention heads.
        heads_dim: Dimension of each attention head.
        use_layernorm: Whether to apply LayerNorm in each LapNet block.
        use_input_bias: Whether to use bias in the input projection.
        use_backbone_bias: Whether to use bias in LapNet layer projections.
        num_local_updates: Number of independent individual-stream residual
            updates between attention blocks.
    """

    nspins: tuple[int, int]
    num_layers: int = 4
    num_heads: int = 4
    heads_dim: int = 64
    use_layernorm: bool = False
    use_input_bias: bool = True
    use_backbone_bias: bool = True
    num_local_updates: int = 2

    def setup(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("LapNet requires at least one layer.")
        attention_dim = self.num_heads * self.heads_dim
        self.input_projection = nn.Dense(
            attention_dim,
            use_bias=self.use_input_bias,
            bias_init=nn.initializers.normal(stddev=1.0),
        )
        self.layers = [
            LapNetLayer(
                num_heads=self.num_heads,
                heads_dim=self.heads_dim,
                use_layernorm=self.use_layernorm,
                use_bias=self.use_backbone_bias,
                num_local_updates=(
                    self.num_local_updates if index < self.num_layers - 1 else 0
                ),
            )
            for index in range(self.num_layers)
        ]

    def individual_stream_block(
        self, h_one: jnp.ndarray, spin: jnp.ndarray
    ) -> tuple[jnp.ndarray, QKStreams]:
        """Compute the initial dense stream and per-layer query/key features.

        Args:
            h_one: One-electron input features of shape `(feature_dim,)`.
            spin: Spin encoding, `+1` for spin-up or `-1` for spin-down.

        Returns:
            Tuple `(hd0, ((q0, k0), ..., (qL-1, kL-1)))` where `hd0` is the
            initial dense stream and each `(qi, ki)` comes from the evolving
            individual stream.
        """
        features = jnp.concatenate([h_one, spin[None]], axis=-1)
        hs = self.input_projection(features)
        hd0 = hs

        qk_streams = []
        for layer in self.layers:
            qk_streams.append(layer.project_qk_stream(hs))
            hs = layer.qk_stream_block(hs)
        return hd0, tuple(qk_streams)

    def dense_block(
        self, stream_features: tuple[jnp.ndarray, QKStreams]
    ) -> jnp.ndarray:
        """Advance the dense stream using precomputed individual-stream Q/K.

        Args:
            stream_features: Tuple `(hd0, qk_streams)` where `hd0` has shape
                `(n_electrons, hidden_width)` and each query/key stream has the
                same leading electron axis.

        Returns:
            Final dense-stream features of shape `(n_electrons, hidden_width)`.

        Raises:
            ValueError: If the number of query/key inputs does not match the
                number of layers.
        """
        dense_stream, qk_streams = stream_features
        if len(qk_streams) != len(self.layers):
            raise ValueError(
                "LapNet dense_block expects one projected query/key input per layer."
            )
        for layer, qk_projection in zip(self.layers, qk_streams, strict=True):
            dense_stream = layer.dense_block(qk_projection, dense_stream)
        return dense_stream

    def __call__(self, h_one: jnp.ndarray) -> jnp.ndarray:
        """Process one-electron features through LapNet layers.

        Args:
            h_one: One-electron features of shape `(n_electrons, feature_dim)`.

        Returns:
            Processed dense-stream features of shape
            `(n_electrons, num_heads * heads_dim)`.

        Raises:
            ValueError: If the input electron count does not match `nspins`.
        """
        n_up, n_down = self.nspins
        n_electrons = n_up + n_down
        if h_one.shape[0] != n_electrons:
            raise ValueError(
                f"Input h_one has {h_one.shape[0]} electrons, "
                f"but nspins={self.nspins} expects {n_electrons} electrons."
            )
        spins = jnp.concatenate([jnp.ones(n_up), -jnp.ones(n_down)])
        stream_features = jax.vmap(self.individual_stream_block)(h_one, spins)
        return self.dense_block(stream_features)
