from pyscf import gto

from lapnet import base_config
from jaqmc.pp.pp_config import get_config as get_ecp_config
from jaqmc.pp.ecp.data import load_ecp_variant as ecpvar


def get_config(input_str):
    symbol, dist, spin, charge, Xup, Xdn, Oup, Odn, ecps = input_str.split(',')

    cfg = base_config.default()
    cfg['ecp'] = get_ecp_config()

    mol = gto.Mole()
    mol.build(
        atom=f'{symbol} 0 0 0; O 0 0 {dist}',
        basis={symbol: 'ccecpccpvdz', 'O': 'ccecpccpvdz'},
        ecp={symbol: ecpvar(symbol, ecps), 'O': 'ccecp'},
        spin=int(spin),
        charge=int(charge),
    )

    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [
        (int(Xup), int(Xdn)),
        (int(Oup), int(Odn)),
    ]

    return cfg