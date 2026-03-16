# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pyscf import gto

from lapnet import base_config
from jaqmc.pp.ph.data import PH_config, extract_L2_data, extract_semilocal_data
from jaqmc.pp.pp_config import get_config as get_ecp_config

@PH_config
def _get_standard_config(input_str):
    symbol, spin = input_str.split(',')
    spin = int(spin)

    cfg = base_config.default()
    cfg['ecp'] = get_ecp_config()
    mol = gto.Mole()
    # Set up molecule
    mol.build(
        atom=f'{symbol} 0 0 0',
        basis={symbol: 'ccecpccpvdz'},
        ecp={symbol: 'ccecp'},
        spin=spin)

    cfg.system.pyscf_mol = mol
    return cfg

def get_config(input_str: str):
    parts = input_str.split(',')

    # Standard PH case, e.g. "Fe,4"
    if len(parts) == 2:
        return _get_standard_config(input_str)

    # Custom Sc L2 case, e.g. "Sc,1,tr2,l2,0"
    elif len(parts) == 5:
        symbol, spin, pp_name, pp_type, charges = (
            parts[0], int(parts[1]), parts[2], parts[3], int(parts[4])
        )

        # make mole of pyscf
        mol = gto.Mole()
        mol.build(
            atom=f"{symbol} 0 0 0",
            basis={symbol: "ccecpccpvdz"},
            ecp={symbol: "ccecp"},
            spin=spin,
            charge=charges,
        )

        cfg = base_config.default()
        cfg.system.pyscf_mol = mol
        cfg["ecp"] = get_ecp_config()
        cfg.system.atom_spin_configs = [(6, 4)]

        # make ph_data
        if symbol == "Sc":
            xml_file = f"{PP_DIRECTORY}/Sc.{pp_name}.xml"
            loc_data = extract_semilocal_data(xml_file=xml_file, l_target="d")
            l2_data = extract_L2_data(xml_file=xml_file)
            if pp_type == "l2":
                ph_data = dict(Sc=(loc_data + 11.0, l2_data))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # register ph_info to config
        cfg.ecp.ph_info = ([(symbol, (0, 0, 0))], ph_data)

        return cfg

    else:
        raise ValueError(
            'Unsupported input format. '
            'Use "symbol,spin" for the standard PH case '
            'or "symbol,spin,pp_name,pp_type,charge" for the custom Sc L2 case.'
        )