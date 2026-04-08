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
    symbol, dist, spin, charge, Xup, Xdn, Sup, Sdn = input_str.split(',')

    dist = float(dist)
    spin = int(spin)
    charge = int(charge)
    Xup = int(Xup)
    Xdn = int(Xdn)
    Sup = int(Sup)
    Sdn = int(Sdn)

    cfg = base_config.default()
    cfg["ecp"] = get_ecp_config()

    mol = gto.Mole()
    mol.build(
        atom=f"{symbol} 0 0 0; S 0 0 {dist}",
        basis={symbol: "ccecpccpvdz", "S": "ccecpccpvdz"},
        ecp={symbol: ecpvar(symbol, "nl"), "S": "ccecp"},
        spin=spin,
        charge=charge,
    )

    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [
        (Xup, Xdn),
        (Sup, Sdn),
    ]
    
    # keep JaQMC electron count consistent with atom_spin_configs
    cfg.system.electrons = (Xup + Sup, Xdn + Sdn)

    cfg.ecp.hph_elements = (symbol,)
    cfg.ecp.ph_elements = ("S",)

    return cfg