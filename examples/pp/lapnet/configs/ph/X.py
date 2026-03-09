"""
Input configuration generator for a single transition-metal atom (TM X)
using the (PH) potential in JaQMC.

This input:
1. Reads pseudopotential data of semilocal and L2 part from XML PH file.
2. Extracts semilocal and L2 radial functions required for the PH formalism.
3. Builds a PySCF Mole object for a single TM atom located at the origin.
4. Constructs the JaQMC configuration including PH pseudopotential data.

The resulting configuration is intended for single-atom transition-metal
that supports hybrid pseudohamiltonian modes.

Returns:
    cfg: JaQMC ConfigDict describing the TM single-atom system with PH pseudopotential.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
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

def extract_L2_data(xml_file):
    """
    Extract the L2 radial function data from a pseudopotential XML file.

    The XML pseudopotential format stores the L2 correction inside the
    following hierarchy:

        L2 → radfunc → data

    The numeric values inside <data> may appear as compact text without
    clear whitespace separators, so a regular expression is used to
    reliably extract the floating-point numbers.

    Args:
        xml_file (str or Path):
            Path to the pseudopotential XML file.

    Returns:
        numpy.ndarray:
            Array containing the L2 radial function values.

    Raises:
        ValueError:
            If the L2/radfunc/data tag cannot be found in the XML file.
        ValueError:
            If no numeric values can be extracted from the data section.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    tag = root.find(".//L2//radfunc//data")
    if tag is None:
        raise ValueError("L2/radfunc/data was not found in the XML file")

    text = tag.text

    tokens = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)

    if len(tokens) == 0:
        raise ValueError("No L2 data values were found (regex extraction returned zero tokens)")

    arr = np.array([float(t) for t in tokens])
    return arr


def parse_compact_data(text):
    """
    Extract floating-point numbers from compact numeric text.

    The input text may contain a sequence of numbers without clear
    delimiters (e.g., no spaces or commas). A regular expression is
    used to identify and extract valid numeric patterns, including
    integers, decimals, and scientific notation.

    Args:
        text (str):
            Raw text containing compact numeric data.

    Returns:
        numpy.ndarray:
            Array of floating-point values extracted from the text.
    """
    tokens = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    return np.array([float(t) for t in tokens])


def extract_semilocal_data(xml_file, l_target="d"):
    """
    Extract semilocal radial potential data for a specified angular momentum
    channel from a pseudopotential XML file.

    The semilocal part of the pseudopotential is stored in the XML structure:

        semilocal → vps(l=...) → radfunc → data

    Each <vps> element corresponds to a specific angular momentum channel
    (e.g., s, p, d). This function searches for the requested channel and
    returns its radial function data.

    Args:
        xml_file (str or Path):
            Path to the pseudopotential XML file.
        l_target (str, optional):
            Angular momentum channel to extract (e.g., "s", "p", "d").

    Returns:
        numpy.ndarray:
            Array containing the semilocal radial potential values.

    Raises:
        RuntimeError:
            If the specified angular momentum channel is not found in the
            semilocal section.
        RuntimeError:
            If the radial function data for the specified channel is missing.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    vps_list = root.findall(".//semilocal//vps")

    for vps in vps_list:
        if vps.get("l") == l_target:
            data_tag = vps.find(".//radfunc//data")
            if data_tag is None:
                raise RuntimeError(f"No radial function data found for vps(l={l_target}).")

            text = data_tag.text
            return parse_compact_data(text)

    raise RuntimeError(f"No semilocal vps entry found for angular momentum l={l_target}.")

def get_config(input_str: str):
    # get element symbol and total spin
    parts = input_str.split(",")
    symbol, spin, pp_name, pp_type, charges = (parts[0], int(parts[1]), parts[2], parts[3], int(parts[4]))

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
