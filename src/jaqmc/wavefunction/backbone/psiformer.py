# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from enum import StrEnum

from flax import linen as nn
from jax import numpy as jnp

__all__ = ["LayerNormMode", "PsiformerBackbone", "PsiformerLayer"]


class LayerNormMode(StrEnum):
    """LayerNorm application mode for Psiformer layers.

    Attributes:
        pre: Apply LayerNorm before attention/MLP blocks (Pre-LN).
            Matches internal_ferminet implementation.
        post: Apply LayerNorm after attention/MLP blocks (Post-LN).
            Matches public FermiNet implementation.
        null: No LayerNorm applied.
    """

    pre = "pre"
    post = "post"
    null = "null"


class PsiformerLayer(nn.Module):
    """Single Psiformer layer combining attention and MLP with residual connections.

    Each layer applies:

    1. Optional LayerNorm (before or after, depending on mode)
    2. Multi-head self-attention with residual connection
    3. Optional LayerNorm (before or after, depending on mode)
    4. MLP with residual connection

    Args:
        num_heads: Number of attention heads.
        heads_dim: Dimension of each attention head.
        mlp_hidden_dims: Hidden dimensions for the MLP block.
        layer_norm_mode: LayerNorm application mode. Options:

            - :attr:`LayerNormMode.pre`: Apply LayerNorm before attention/MLP
              blocks (Pre-LN, matches internal_ferminet). Default.
            - :attr:`LayerNormMode.post`: Apply LayerNorm after attention/MLP
              blocks (Post-LN, matches public FermiNet).
            - :attr:`LayerNormMode.null`: No LayerNorm applied.

        with_bias: Whether to use bias in attention QKV projections.
    """

    num_heads: int
    heads_dim: int
    mlp_hidden_dims: Sequence[int]
    layer_norm_mode: LayerNormMode = LayerNormMode.pre
    with_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply one Psiformer layer.

        Args:
            x: Input features of shape ``(n_electrons, attention_dim)``.

        Returns:
            Output features of shape ``(n_electrons, attention_dim)``.
        """
        attention_dim = self.num_heads * self.heads_dim

        x_in = x
        if self.layer_norm_mode == LayerNormMode.pre:
            x_in = nn.LayerNorm(epsilon=1e-5)(x)

        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=attention_dim,
            out_features=attention_dim,
            deterministic=True,
            use_bias=self.with_bias,
        )(x_in)
        x = x + attn_out

        if self.layer_norm_mode == LayerNormMode.pre:
            mlp_out = nn.LayerNorm(epsilon=1e-5)(x)
        elif self.layer_norm_mode == LayerNormMode.post:
            mlp_out = x = nn.LayerNorm(epsilon=1e-5)(x)
        elif self.layer_norm_mode == LayerNormMode.null:
            mlp_out = x

        for dim in self.mlp_hidden_dims:
            mlp_out = nn.tanh(nn.Dense(dim)(mlp_out))
        mlp_out = nn.tanh(nn.Dense(attention_dim)(mlp_out))
        x = x + mlp_out

        if self.layer_norm_mode == LayerNormMode.post:
            x = nn.LayerNorm(epsilon=1e-5)(x)
        return x


class PsiformerBackbone(nn.Module):
    """Self-attention backbone for molecular wavefunctions.

    Psiformer processes one-electron features through multiple self-attention
    layers, modeling electron-electron interactions implicitly through
    attention rather than explicit two-electron feature streams.

    The architecture:

    1. Concatenates spin encoding to input features
    2. Projects to attention dimension
    3. Applies ``num_layers`` PsiformerLayer blocks
    4. Outputs processed one-electron features

    Args:
        nspins: Tuple of (num_spin_up, num_spin_down) electrons.
        num_layers: Number of Psiformer layers.
        num_heads: Number of attention heads.
        heads_dim: Dimension of each attention head.
        mlp_hidden_dims: Hidden dimensions for MLP blocks.
        layer_norm_mode: LayerNorm application mode. Options:

            - :attr:`LayerNormMode.pre`: Apply LayerNorm before attention/MLP
              blocks (Pre-LN, matches internal_ferminet). Default.
            - :attr:`LayerNormMode.post`: Apply LayerNorm after attention/MLP
              blocks (Post-LN, matches public FermiNet).
            - :attr:`LayerNormMode.null`: No LayerNorm applied.

        with_bias: Whether to use bias in attention QKV projections.
        input_bias: Whether to use bias in the input projection layer.
    """

    nspins: tuple[int, int]
    num_layers: int = 2
    num_heads: int = 4
    heads_dim: int = 64
    mlp_hidden_dims: Sequence[int] = (256,)
    layer_norm_mode: LayerNormMode = LayerNormMode.pre
    with_bias: bool = True
    input_bias: bool = True

    @nn.compact
    def __call__(self, h_one: jnp.ndarray) -> jnp.ndarray:
        """Process one-electron features through self-attention layers.

        Args:
            h_one: One-electron features of shape ``(n_electrons, feature_dim)``.
                Typically atom-electron features from
                :class:`~jaqmc.wavefunction.input.atomic.MoleculeFeatures`.

        Returns:
            Processed features of shape ``(n_electrons, num_heads * heads_dim)``.

        Raises:
            ValueError: If ``h_one.shape[0] != sum(nspins)``.
        """
        n_up, n_down = self.nspins
        n_electrons = n_up + n_down
        if h_one.shape[0] != n_electrons:
            raise ValueError(
                f"Input h_one has {h_one.shape[0]} electrons, "
                f"but nspins={self.nspins} expects {n_electrons} electrons."
            )

        attention_dim = self.num_heads * self.heads_dim

        # Create spin encoding: +1 for spin-up, -1 for spin-down
        spins = jnp.concatenate([jnp.ones(n_up), -jnp.ones(n_down)])

        # Concatenate spin feature for permutation equivariance
        features = jnp.concatenate([h_one, spins[..., None]], axis=-1)

        # Embed to attention dimension
        x = nn.Dense(attention_dim, use_bias=self.input_bias)(features)

        # Apply Psiformer layers
        for _ in range(self.num_layers):
            x = PsiformerLayer(
                num_heads=self.num_heads,
                heads_dim=self.heads_dim,
                mlp_hidden_dims=self.mlp_hidden_dims,
                layer_norm_mode=self.layer_norm_mode,
                with_bias=self.with_bias,
            )(x)

        return x
