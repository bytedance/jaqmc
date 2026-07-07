# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""PH reference data loader.

This module loads pseudo-Hamiltonian (PH) reference tables from the XML
files bundled inside this package at ``raw_data/{TM,MG}/``. The
tabulated data follows the ``r * V`` semilocal convention used by the
original PH library accompanying Bennett et al. (2022) and its
locality-error-free follow-up Ichibha et al. (2023) for 3d transition
metals, and Fu et al. (2025) for sulfur. PH valence electron counts and
the supported-element set live in :mod:`jaqmc.utils.atomic.pp` (the
pseudopotential vocabulary layer).

References:
    Bennett, M. C., Reboredo, F. A., Mitas, L., Krogel, J. T.,
    "High Accuracy Transition Metal Effective Cores for the Many-Body
    Diffusion Monte Carlo Method,"
    *J. Chem. Theory Comput.* **18**, 828-839 (2022).
    DOI: `10.1021/acs.jctc.1c00992
    <https://doi.org/10.1021/acs.jctc.1c00992>`_.

    Ichibha, T., Nikaido, Y., Bennett, M. C., Krogel, J. T., Hongo, K.,
    Maezono, R., Reboredo, F. A.,
    "Locality error free effective core potentials for 3d transition
    metal elements developed for the diffusion Monte Carlo method,"
    *J. Chem. Phys.* **159**, 164114 (2023).
    DOI: `10.1063/5.0175381 <https://doi.org/10.1063/5.0175381>`_.

    Fu, W. *et al.*, "Local Pseudopotential Unlocks the True Potential
    of Neural Network-based Quantum Monte Carlo,"
    `arXiv:2505.19909 <https://arxiv.org/abs/2505.19909>`_ (2025).
"""

import xml.etree.ElementTree as ET
from functools import cache
from importlib import resources

import numpy as np

from jaqmc.utils.atomic import elements
from jaqmc.utils.atomic.pp import (
    PP_PH,
    SUPPORTED_PH_ELEMENTS,
    core_electrons_by_pp,
)

__all__ = ["load_ph_element_data"]

_ANGULAR_SYMBOLS = ("s", "p", "d", "f", "g", "h", "i", "j", "k", "l")

# Per-element XML loading metadata: ``(subdirectory, use_hf)``.
# ``subdirectory`` is the ``raw_data/`` subfolder ("TM" or "MG"); ``use_hf``
# selects between the ``.hf.xml`` and ``.cc.xml`` filename variants. Both
# are XML-loading details unrelated to the PH valence count, which lives in
# :mod:`jaqmc.utils.atomic.pp`.
_PH_XML_INFO: dict[str, tuple[str, bool]] = {
    "Cr": ("TM", False),
    "Mn": ("TM", True),
    "Fe": ("TM", False),
    "Co": ("TM", False),
    "Ni": ("TM", True),
    "Cu": ("TM", True),
    "Zn": ("TM", False),
    "S": ("MG", False),
}
_RAW_DATA_DIR = resources.files("jaqmc.estimator.ph").joinpath("raw_data")


def load_ph_element_data(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    r"""Load the PH radial tables for a supported element.

    Reads the bundled XML file for ``symbol``, reconstructs the paper-form
    :math:`(\tilde v_{\mathrm{loc}}, v_{L^2})` channels (see
    :func:`_reconstruct_paper_ph_data`), and returns them in the XML's
    :math:`r \cdot V` tabulation convention.

    **Local-channel sign convention.** The returned ``loc_data`` table is
    :math:`r \cdot \tilde v_{\mathrm{loc}}(r) + Z_a`, i.e. the bare paper-form
    local channel plus the effective PH valence charge ``Z_a`` computed as
    atomic number minus the PH core count from
    :func:`jaqmc.utils.atomic.pp.core_electrons_by_pp`. The estimator
    consumes this as ``loc_data(r) / r``, which decays to zero at large
    ``r`` and is bounded near the origin. Because the bare electron-nucleus
    Coulomb ``-Z_a / r`` is supplied separately by ``potential_energy``,
    summing ``energy:potential + energy:ph`` recovers the paper-form
    ``sum_e sum_a tilde_v_loc(r_ea)`` without any masking between the two
    estimators. See ``PHEnergy._evaluate_zero_order_term`` for the
    consumer side of this contract.

    **Radial grid.** All bundled XML files use a fixed linear grid
    :math:`r \in [0, 10]` bohr with 10001 points. Callers reconstruct the
    grid via ``jnp.linspace(0.0, 10.0, data.shape[0])``. XML files with a
    different grid layout (different ``ri``/``rf``/``npts``) are not
    supported by this loader.

    The underlying parser caches its output and marks the returned arrays
    read-only; this function returns fresh copies so callers may mutate
    them freely.

    Args:
        symbol: Element symbol (must be in
            :data:`jaqmc.utils.atomic.pp.SUPPORTED_PH_ELEMENTS`).

    Returns:
        Tuple ``(loc_data, l2_data)``, each with shape ``(10001,)``:
        ``loc_data`` is :math:`r \cdot \tilde v_{\mathrm{loc}}(r) + Z_a`,
        and ``l2_data`` is :math:`r \cdot v_{L^2}(r)`.

    Raises:
        ValueError: If ``symbol`` is not a supported PH element.
    """
    if symbol not in SUPPORTED_PH_ELEMENTS:
        raise ValueError(f"unsupported PH element: {symbol}")

    resource_path = _element_resource_path(symbol)
    atomic_number = elements.from_symbol[symbol].atomic_number
    charge = atomic_number - core_electrons_by_pp(symbol, PP_PH)
    loc_data, l2_data = _load_ph_data_from_xml(resource_path, charge)
    return loc_data.copy(), l2_data.copy()


def _element_resource_path(symbol: str) -> str:
    try:
        subdir, use_hf = _PH_XML_INFO[symbol]
    except KeyError as err:
        raise ValueError(f"unsupported PH element: {symbol}") from err
    filename = f"{symbol}.{'hf' if use_hf else 'cc'}.xml"
    return f"{subdir}/{filename}"


@cache
def _load_ph_data_from_xml(
    resource_path: str, charge: float
) -> tuple[np.ndarray, np.ndarray]:
    xml_file = _RAW_DATA_DIR.joinpath(resource_path)
    with xml_file.open("rb") as fh:
        tree = ET.parse(fh)
    root = tree.getroot()

    semilocal = root.find("semilocal")
    if semilocal is None:
        raise ValueError(f"missing semilocal section in XML file: {xml_file}")

    loc_data, l2_data = _reconstruct_paper_ph_data(semilocal)

    loc_data = loc_data + charge
    loc_data.setflags(write=False)
    l2_data.setflags(write=False)
    return loc_data, l2_data


def _extract_semilocal_channel(semilocal: ET.Element, l_target: str) -> np.ndarray:
    for vps in semilocal.findall("vps"):
        if vps.get("l") == l_target:
            data_tag = vps.find("radfunc/data")
            if data_tag is None or data_tag.text is None:
                raise ValueError(f"missing PH data for l={l_target}")
            return _parse_compact_data(data_tag.text)
    raise ValueError(f"missing PH data for l={l_target}")


def _reconstruct_paper_ph_data(semilocal: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct ``(tilde_v_loc, v_L2)`` from a semilocal PH(M) XML block.

    The 2022 pseudo-Hamiltonian paper represents the truncated semilocal form
    ``PH(M)`` as

    ``<l|v_PH(M)|l> = tilde_v_loc + l(l+1) v_L2`` for ``l <= M``,

    while the semilocal local channel is

    ``v_loc^SL = tilde_v_loc + M(M+1) v_L2``.

    The XML stores the semilocal channels in ``r * V`` form, so the same
    algebra applies pointwise to the tabulated arrays.

    Returns:
        Tuple ``(tilde_v_loc, v_L2)`` in the tabulated ``r * V`` convention.

    Raises:
        ValueError: If the XML is missing the local angular momentum metadata or
            uses an unsupported angular channel index.
    """
    local_l = semilocal.get("l-local")
    if local_l is None:
        raise ValueError("missing l-local attribute in PH semilocal XML")
    local_l_index = int(local_l)
    if local_l_index >= len(_ANGULAR_SYMBOLS):
        raise ValueError(f"unsupported PH l-local index: {local_l_index}")

    channels = {
        symbol: _extract_semilocal_channel(semilocal, symbol)
        for symbol in _ANGULAR_SYMBOLS[: local_l_index + 1]
    }
    local_symbol = _ANGULAR_SYMBOLS[local_l_index]
    local_channel = channels[local_symbol]

    if local_l_index == 0:
        return local_channel.copy(), np.zeros_like(local_channel)

    l2_candidates = []
    for l_index, symbol in enumerate(_ANGULAR_SYMBOLS[:local_l_index]):
        denominator = l_index * (l_index + 1) - local_l_index * (local_l_index + 1)
        l2_candidates.append((channels[symbol] - local_channel) / denominator)
    l2_data = np.mean(np.stack(l2_candidates, axis=0), axis=0)
    loc_data = local_channel - local_l_index * (local_l_index + 1) * l2_data
    return loc_data, l2_data


def _parse_compact_data(text: str) -> np.ndarray:
    tokens = np.fromstring(text, sep=" ")
    if tokens.size == 0:
        tokens = np.array(
            [
                float(token)
                for token in text.split()
                if token and token not in {"\n", "\t"}
            ]
        )
    return tokens
