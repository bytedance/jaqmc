from jaqmc.pp.pp_config import get_config as get_ecp_config
from lapnet import base_config
from pyscf import gto

from pathlib import Path

"""
Resolve the path to the pseudopotential directory inside the JaQMC repository.
Path(__file__) gives the location of this Python file.
.resolve() converts it to an absolute path.
.parents[2] moves two directories up from this file's location

Then we append "jaqmc/pseudopotentials", which contains the PH
pseudopotential files (XML).
"""

PP_DIRECTORY = Path(__file__).resolve().parents[2] / "jaqmc" / "pseudopotentials"


def read_text(path):
    """
    Read the entire contents of a text file.

    This utility is mainly used to load pseudopotential files
    (e.g., NWChem ECP files) stored in the JaQMC pseudopotential directory.

    Args:
        path: Path to the file to be read.

    Returns:
        str: The full text content of the file.
    """
    with open(path, "r") as f:
        return f.read()

def get_config(input_str: str):
    # get element symbol and total spin
    parts = input_str.split(",")
    symbol, spin, pp_name = str(parts[0]), int(parts[1]), str(parts[2])

    # make ecp data
    ecp_path = f"{PP_DIRECTORY}/{symbol}.{pp_name}.nwchem"
    if Path(ecp_path).exists():
        ecp_text = read_text(ecp_path)
        ecp_data = gto.basis.parse_ecp(ecp_text)
    else:
        raise FileNotFoundError(f"File not found: {ecp_path}")

   # make mole of pyscf
    mol = gto.Mole()
    mol.build(
        atom=f"{symbol} 0 0 0 ; O 0 0 1.664", # X-O distance
        # ccsdt using 1.668 A
        basis={symbol: "ccecpccpvdz", "O": "ccpvdz"},
        ecp={symbol: ecp_data},
        spin=spin,
    )

    # make mole of nnqmc
    cfg = base_config.default()
    cfg.system.pyscf_mol = mol
    cfg["ecp"] = get_ecp_config()
    cfg.system.atom_spin_configs = [(6, 5), (4, 4)]   # Manually set the number of spin up and spin down electrons

    return cfg
