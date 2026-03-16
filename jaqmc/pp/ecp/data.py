from pathlib import Path
from pyscf import gto

SC_FAMILY = "0.21_187"
RAW_DATA_DIR = Path(__file__).resolve().parent / "raw_data"

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