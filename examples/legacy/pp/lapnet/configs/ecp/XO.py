# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.pp_config import get_config as get_ecp_config
from jaqmc.pp.ecp.data import load_ecp_variant as ecpvar


def get_config(input_str):

    """
    The inputs are expected to be in one of the following formats:
    symbol: the chemical symbol of the element (e.g., "Sc")
    spin: the total spin of the system (e.g., "1")
    charge: the total charge of the system (e.g., "0")
    Xup: the number of spin-up electrons in the valence shell (e.g., "6")
    Xdn: the number of spin-down electrons in the valence shell (e.g., "5")
    Oup: the number of spin-up electrons in the oxygen atom (e.g., "4")
    Odn: the number of spin-down electrons in the oxygen atom (e.g., "4")
    ecps: a string specifying the ECP variant to use (e.g., "ccecp" from PySCF or "tr2" from user's custom ECPs)
    """

    symbol, dist, spin, charge, Xup, Xdn, Oup, Odn, ecps = input_str.split(',')

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
        ecp={symbol: ecpvar(symbol, ecps)},
        spin=int(spin),
        charge=int(charge),
    )

    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [
        (Xup, Xdn),
        (Oup, Odn),
    ]

    # keep JaQMC electron count consistent with atom_spin_configs
    cfg.system.electrons = (Xup + Oup, Xdn + Odn)

    return cfg
