# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.ph.data import PH_config
from jaqmc.pp.pp_config import get_config as get_ecp_config
from jaqmc.pp.ecp.data import load_ecp_variant as ecpvar

@PH_config
def get_config(input_str):
    symbol, spin, charge, Xup, Xdn = input_str.split(',')

    spin = int(spin)
    charge = int(charge)
    Xup = int(Xup)
    Xdn = int(Xdn)

    cfg = base_config.default()
    cfg["ecp"] = get_ecp_config()

    mol = gto.Mole()
    mol.build(
        atom=f"{symbol} 0 0 0",
        basis={symbol: "ccecpccpvdz"},
        ecp={symbol: ecpvar(symbol, "nl")},
        spin=spin,
        charge=charge,
    )

    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [(Xup, Xdn)]

    cfg.ecp.hph_elements = (symbol,)

    return cfg