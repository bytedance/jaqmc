# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import ml_collections

def get_config():
  return ml_collections.ConfigDict({
        'use_ecp_optim':False, # if use ecp optim
        'ecp_select_core':{
          # When calculating ECP (Effective Core Potential),
          # in principle, it is necessary to integrate over
          # all atomic nuclei. However, due to the form of the integration,
          # the integral for nuclei that are far from the electrons is zero.
          # Utilizing this property,
          # one can choose to integrate only the nuclei that are close to the electrons.
          # The parameter max_core determines the number of nearby nuclei to be selected for integration.
          # you can choose the max_core by ecp cutoff range
          # max_core = 2 can meet the needs of most molecular systems
          'max_core': 2
        },
        'ecp_quadrature_id': 'icosahedron_12',# quadrature rule
        # PH related information.
        # NOT supposed to be specified over command line.
        # When using the "PH_config" decorator in the config file,
        # this config will be updated automatically.
        'ph_info': None,
        # None (the default value) means all available ph elements.
        # If you want to use a subset of ph elements, you can specify them as a set in the config file.
        # For example, if you want to use Cr and Mn, you can specify it as {'Cr', 'Mn'}.
        # WARNING: Customization through command line is NOT supported.
        'ph_elements': None,
        # PH rv_type could be 'spline' or 'linear'.
        # Not much difference from our tests. We use 'spline' as default since
        # that's (what we believe) what's used when PH is constructed.
        # That said, we strongly disencourage mixing the results from calculations
        # with 'spline' and 'linear'. If you choose 'spline', stick with it in
        # all your calculations. Comparing 'spline' result and 'linear' one
        # is technically not apple-to-apple comparision.
        'ph_rv_type': 'spline'
      },
  )
