# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.ph.data import PH_config, load_sc_ph_data
from jaqmc.pp.pp_config import get_config as get_ecp_config
from jaqmc.pp.ecp.data import load_ecp_variant as ecpvar


@PH_config
def _get_standard_config(input_str):
    symbol, spin = input_str.split(',')
    spin = int(spin)

    cfg = base_config.default()
    cfg['ecp'] = get_ecp_config()

    mol = gto.Mole()
    mol.build(
        atom=f'{symbol} 0 0 0',
        basis={symbol: 'ccecpccpvdz'},
        ecp={symbol: 'ccecp'},
        spin=spin,
    )

    cfg.system.pyscf_mol = mol
    return cfg


def get_config(input_str: str):

    parts = input_str.split(',')

    # Case 1: original PH
    if len(parts) == 2:
        return _get_standard_config(input_str)

    # Case 2/3: Sc L2 or hybrid
    elif len(parts) == 6:

        symbol, spin, pp_type, charge, Xup, Xdn = parts

        spin = int(spin)
        charge = int(charge)
        Xup = int(Xup)
        Xdn = int(Xdn)

        if symbol != "Sc":
            raise NotImplementedError("Custom PH case only implemented for Sc")

        if pp_type not in ("l2", "hybrid"):
            raise NotImplementedError(f"Unsupported pp_type: {pp_type}")

        cfg = base_config.default()
        cfg["ecp"] = get_ecp_config()

        # choose ECP
        if pp_type == "l2":
            ecp_data = "ccecp"
        else:
            ecp_data = ecpvar(symbol, "nl")

        mol = gto.Mole()
        mol.build(
            atom=f"{symbol} 0 0 0",
            basis={symbol: "ccecpccpvdz"},
            ecp={symbol: ecp_data},
            spin=spin,
            charge=charge,
        )

        cfg.system.pyscf_mol = mol
        cfg.system.atom_spin_configs = [(Xup, Xdn)]

        ph_data = load_sc_ph_data()

        cfg.ecp.ph_info = ([(symbol, (0, 0, 0))], ph_data)
        cfg.ecp.ph_mode = pp_type

        return cfg

    else:
        raise ValueError(
            'Unsupported input format.\n'
            'Use "symbol,spin" for original PH\n'
            'or "Sc,spin,pp_type,charge,Xup,Xdn" for custom Sc PH.'
        )