# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.ph.data import PH_config, load_sc_ph_data
from jaqmc.pp.pp_config import get_config as get_ecp_config

@PH_config
def get_config(input_str):
    symbol, spin, charge, Xup, Xdn = input_str.split(',')

    spin = int(spin)
    charge = int(charge)
    Xup = int(Xup)
    Xdn = int(Xdn)

    cfg = base_config.default()
    cfg['ecp'] = get_ecp_config()

    mol = gto.Mole()
    mol.build(
        atom=f'{symbol} 0 0 0',
        basis={symbol: 'ccecpccpvdz'},
        ecp={symbol: 'ccecp'},
        spin=spin,
        charge=charge
    )

    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [(Xup, Xdn)]

    cfg.ecp.ph_elements = (symbol,)

    return cfg
