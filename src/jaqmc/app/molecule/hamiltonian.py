# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jax import numpy as jnp

from jaqmc.geometry import obc


def potential_energy(params, data, prev_walker_stats, state, rngs):
    del params, rngs, prev_walker_stats
    nelec = data.electrons.shape[0]
    natom = data.atoms.shape[0]
    r_ae = obc.pair_displacements_between(data.electrons, data.atoms)[1]
    r_ee = obc.pair_displacements_within(data.electrons)[1] + jnp.eye(nelec)
    r_aa = obc.pair_displacements_within(data.atoms)[1] + jnp.eye(natom)
    return {
        "energy:potential": (
            jnp.sum(-jnp.ones(nelec)[:, None] * data.charges / r_ae)
            + jnp.sum(jnp.triu(1 / r_ee, k=1))
            + jnp.sum(jnp.triu(data.charges * data.charges[:, None] / r_aa, k=1))
        )
    }, state
