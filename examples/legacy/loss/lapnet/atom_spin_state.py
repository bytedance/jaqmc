# Copyright (c) 2024-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from lapnet import base_config
from lapnet.utils import system


def atom_set(cfg, spin):
    atom = system.Atom(cfg.system.atom)
    cfg.system.molecule = [atom]

    if (atom.charge - spin) % 2 != 0.:
        raise ValueError(f"Wrongly assign spin! Difference between spin-up and -down cannot be {spin} for {atom.charge} electrons!")

    alpha = int(spin + (atom.charge - spin) // 2)
    beta = int((atom.charge - spin) // 2)
    cfg.system.electrons = [alpha, beta]
    cfg.system.atom_spin_configs = [[alpha, beta],]
    return cfg


def get_config(input_str):
    name, spin = input_str.split(',')
    spin = int(spin)
    cfg = base_config.default()
    cfg.system.atom = name
    cfg = atom_set(cfg, spin)
    return cfg
