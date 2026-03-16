# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.ph.data import PH_config, load_sc_ph_data
from jaqmc.pp.pp_config import get_config as get_ecp_config
from jaqmc.pp.ecp.data import load_ecp_variant as ecpvar


@PH_config
def _get_standard_config(input_str):
    symbol, dist, spin, Xup, Xdn, Yup, Ydn = input_str.split(',')

    cfg = base_config.default()
    cfg['ecp'] = get_ecp_config()

    mol = gto.Mole()

    mol.build(
        atom=f'{symbol} 0 0 0; S 0 0 {dist}',
        basis={symbol: 'ccecpccpvdz', 'S': 'ccecpccpvdz'},
        ecp={symbol: 'ccecp', 'S': 'ccecp'},
        spin=int(spin),
    )

    cfg.system.pyscf_mol = mol
    cfg.system.atom_spin_configs = [
        (int(Xup), int(Xdn)),
        (int(Yup), int(Ydn)),
    ]

    cfg.ecp.ph_elements = (symbol, 'S')

    return cfg


def get_config(input_str: str):

    parts = input_str.split(',')

    # Case 1: original PH
    # symbol,dist,spin,Xup,Xdn,Yup,Ydn
    if len(parts) == 7:
        return _get_standard_config(input_str)

    # Case 2/3: custom ScO PH l2 / hybrid
    # Sc,dist,spin,pp_type,charge,Xup,Xdn,Oup,Odn
    elif len(parts) == 9:

        symbol, dist, spin, pp_type, charge, Xup, Xdn, Oup, Odn = parts

        spin = int(spin)
        charge = int(charge)

        Xup = int(Xup)
        Xdn = int(Xdn)
        Oup = int(Oup)
        Odn = int(Odn)

        if symbol != "Sc":
            raise NotImplementedError("Custom PH XO case only implemented for Sc")

        if pp_type not in ("l2", "hybrid"):
            raise NotImplementedError(f"Unsupported pp_type: {pp_type}")

        cfg = base_config.default()
        cfg["ecp"] = get_ecp_config()

        if pp_type == "l2":
            ecp_data = "ccecp"
        else:
            ecp_data = ecpvar(symbol, "nl")

        mol = gto.Mole()
        mol.build(
            atom=f"{symbol} 0 0 0; O 0 0 {dist}",
            basis={symbol: "ccecpccpvdz", "O": "ccpvdz"},
            ecp={symbol: ecp_data},
            spin=spin,
            charge=charge,
        )

        cfg.system.pyscf_mol = mol

        cfg.system.atom_spin_configs = [
            (Xup, Xdn),
            (Oup, Odn),
        ]

        cfg.system.electrons = (Xup + Oup, Xdn + Odn)

        ph_data = load_sc_ph_data()

        cfg.ecp.ph_info = ([("Sc", (0, 0, 0))], ph_data)
        cfg.ecp.ph_mode = pp_type
        cfg.ecp.ph_elements = ("Sc",)

        return cfg

    else:
        raise ValueError(
            'Unsupported input format.\n'
            'Use "symbol,dist,spin,Xup,Xdn,Yup,Ydn" for original PH XO\n'
            'or "Sc,dist,spin,pp_type,charge,Xup,Xdn,Oup,Odn" for custom ScO PH.'
        )