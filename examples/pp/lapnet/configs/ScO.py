# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.ph.data import PH_config, load_sc_ph_data, gen_ph_info
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

    # position and data for Sc
    sc_ph_atom_pos = [(symbol, (0.0, 0.0, 0.0))]
    sc_ph_data = load_sc_ph_data()

    # position and data for S
    s_ph_atom_pos, s_ph_data = gen_ph_info(
        mol._atom,
        ph_elements=("S",),
    )
    
    # combine Sc and S PH info
    ph_atom_pos = sc_ph_atom_pos + s_ph_atom_pos
    ph_data = {}
    ph_data.update(s_ph_data)
    ph_data.update(sc_ph_data)

    cfg.ecp.ph_info = (ph_atom_pos, ph_data)
    cfg.ecp.ph_mode = "hybrid"
    cfg.ecp.ph_elements = (symbol, "S")
    
    # specify which element(s) use the hybrid PH mode
    cfg.ecp.hybrid_elements = ("Sc",)

    return cfg