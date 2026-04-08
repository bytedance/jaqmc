# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.ph.data import PH_config, load_sc_ph_data
from jaqmc.pp.pp_config import get_config as get_ecp_config

@PH_config
def get_config(input_str):
    symbol, dist, spin, charge, Xup, Xdn, Oup, Odn = input_str.split(',')

    spin = int(spin)
    charge = int(charge)
    Xup = int(Xup)
    Xdn = int(Xdn)
    Oup = int(Oup)
    Odn = int(Odn)

    cfg = base_config.default()
    cfg['ecp'] = get_ecp_config()
    
    mol = gto.Mole()
    mol.build(
        atom=f'{symbol} 0 0 0; O 0 0 {dist}',
        basis={symbol: 'ccecpccpvdz', 'O': 'ccpvdz'},
        ecp={symbol: 'ccecp'},
        spin=spin,
        charge=charge)

    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [
        (Xup, Xdn),
        (Oup, Odn),
    ]
    
    # keep JaQMC electron count consistent with atom_spin_configs
    cfg.system.electrons = (Xup + Oup, Xdn + Odn)

    cfg.ecp.ph_elements = (symbol,)

    return cfg