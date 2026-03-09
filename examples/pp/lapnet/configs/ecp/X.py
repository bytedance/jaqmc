# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.pp_config import get_config as get_ecp_config

def get_config(input_str):
    # get element symbol and total spin
    parts = input_str.split(",")
    symbol, spin = str(parts[0]), int(parts[1])

    # make mole of pyscf
    mol = gto.Mole()
    mol.build(
        atom=f"{symbol} 0 0 0",
        basis={symbol: "ccpvdz"},
        spin=spin,
    )

    cfg = base_config.default()
    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [(4, 2)] # manually set the spin up and spin down electrons
    cfg["ecp"] = get_ecp_config()
    return cfg