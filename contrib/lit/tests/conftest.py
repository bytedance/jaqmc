# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Configure enough host devices for the data-parallel test cases."""

import os

_HOST_DEVICE_FLAG = "--xla_force_host_platform_device_count=4"
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} {_HOST_DEVICE_FLAG}".strip()
