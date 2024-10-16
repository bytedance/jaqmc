# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Loss configurations.
'''

from jaqmc.loss.spin_penalty import DEFAULT_SPIN_PENALTY_CONFIG

def get_config():
    return {'enforce_spin': DEFAULT_SPIN_PENALTY_CONFIG}
