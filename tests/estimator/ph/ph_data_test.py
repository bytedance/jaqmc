# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from jaqmc.estimator.ph.data import load_ph_element_data
from jaqmc.utils.atomic import SUPPORTED_PH_ELEMENTS


@pytest.mark.parametrize("symbol", sorted(SUPPORTED_PH_ELEMENTS))
def test_all_supported_ph_elements_load(symbol):
    loc_data, l2_data = load_ph_element_data(symbol)
    assert loc_data.shape
    assert l2_data.shape
    assert loc_data.shape == l2_data.shape
    assert np.isfinite(loc_data).all()
    assert np.isfinite(l2_data).all()
