# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.pp_config import get_config as get_ecp_config
from jaqmc.pp.ecp.data import load_ecp_variant as ecpvar

def get_config(input_str):
    """
    The inputs are expected to be in one of the following formats:
    symbol: the chemical symbol of the element (e.g., "Sc" or "O")
    spin: the total spin of the system (e.g., "2")
    charge: the total charge of the system (e.g., "0")
    Xup: the number of spin-up electrons in the valence shell (e.g., "4")
    Xdn: the number of spin-down electrons in the valence shell (e.g.,"2")
    ecps: a string specifying the ECP variant to use (e.g., "ccecp" or "tr2")
    """

    symbol, spin, charge, Xup, Xdn, ecps = input_str.split(',')

    cfg = base_config.default()
    cfg['ecp'] = get_ecp_config()

    mol = gto.Mole()

    build_kwargs = dict(
        atom=f'{symbol} 0 0 0',
        basis={symbol: 'ccecpccpvdz'},
        spin=int(spin),
        charge=int(charge),
    )

    # Keep O as all-electron
    if symbol != 'O':
        build_kwargs['ecp'] = {symbol: ecpvar(symbol, ecps)}

    mol.build(**build_kwargs)

    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [(int(Xup), int(Xdn))]
    return cfg