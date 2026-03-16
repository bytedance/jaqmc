from importlib import resources
from pyscf import gto

raw_package = "jaqmc.pp.ecp.raw_data"
Sc_family = "0.21_187"

def load_ecp_variant(symbol: str, ecp_variant: str):
    symbol = symbol.strip().capitalize()
    ecp_variant = ecp_variant.strip()

    if symbol == "Sc":
        if ecp_variant.lower() == "ccecp":
            return "ccecp"

        filename = f"{symbol}.{Sc_family}_{ecp_variant}.nwchem"
        resource = resources.files(raw_package).joinpath(filename)

        if not resource.is_file():
            raise FileNotFoundError(f"ECP file not found for Sc: {filename}")

        ecp_text = resource.read_text()
        return gto.basis.parse_ecp(ecp_text, symb=symbol)

    return ecp_variant