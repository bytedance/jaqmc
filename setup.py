# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'attrs',
    'chex',
    'jax',
    'jaxlib',
    'ml-collections',
    'optax',
    'numpy',
    'pandas',
    'pyscf',
    'scipy',
]

setup(
    name='jaqmc',
    version='0.0.1',
    description='JAX accelerated Quantum Monte Carlo',
    author='ByteDance, AI-Lab, Research',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
)
