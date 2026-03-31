# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.ph.data import PH_config
from jaqmc.pp.pp_config import get_config as get_ecp_config


@PH_config
def get_config(input_str):
    symbol, dist, spin, charge, Xup, Xdn, Yup, Ydn = input_str.split(',')

    spin = int(spin)
    charge = int(charge)
    Xup = int(Xup)
    Xdn = int(Xdn)
    Yup = int(Yup)
    Ydn = int(Ydn)

    cfg = base_config.default()
    cfg['ecp'] = get_ecp_config()
    mol = gto.Mole()

    # Set up molecule
    mol.build(
        atom=f'{symbol} 0 0 0; O 0 0 {dist}',
        basis={symbol: 'ccecpccpvdz', 'O': 'ccecpccpvdz'},
        ecp={symbol: 'ccecp', 'O': 'ccecp'},
        spin=spin,
        charge=charge)

    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [
        (Xup, Xdn),
        (Yup, Ydn),
    ]
    cfg.ecp.ph_elements = (symbol, 'O')

    return cfg