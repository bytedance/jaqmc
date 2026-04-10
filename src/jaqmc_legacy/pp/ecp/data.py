# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from pyscf import gto

SC_FAMILY = "0.21_187"
RAW_DATA_DIR = Path(__file__).resolve().parent / "raw_data"

"""
Load ECP for a given element and variant.

For Sc:
- If 'ccecp' → use PySCF built-in keyword
- Otherwise → load custom ECP from local NWChem file

For other elements:
- Return the variant as-is (assumed PySCF keyword)

Parameters
----------
symbol : str
    Atomic symbol (e.g., 'Sc')
ecp_variant : str
    ECP name or variant (e.g., 'ccecp', 'tr2')

Returns
-------
str or dict
    PySCF ECP input (keyword or parsed object)
"""

def load_ecp_variant(symbol: str, ecp_variant: str):
    symbol = symbol.strip().capitalize()
    ecp_variant = ecp_variant.strip().lower()

    if symbol == "Sc":
        if ecp_variant == "ccecp":
            return "ccecp"

        filename = f"{symbol}.{SC_FAMILY}_{ecp_variant}.nwchem"
        path = RAW_DATA_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"ECP file not found for Sc variant '{ecp_variant}': {path}"
            )

        ecp_text = path.read_text()
        return gto.basis.parse_ecp(ecp_text, symb=symbol)

    return ecp_variant
