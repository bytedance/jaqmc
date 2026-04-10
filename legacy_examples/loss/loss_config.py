# Copyright (c) 2022-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

'''
Loss configurations.
'''

from jaqmc.loss.spin_penalty import DEFAULT_SPIN_PENALTY_CONFIG

def get_config():
    return {'enforce_spin': DEFAULT_SPIN_PENALTY_CONFIG}
