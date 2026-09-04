# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from jaqmc_contrib_lit.checkpoint import LITCheckpointManager


def test_lit_checkpoint_round_trip_is_atomic_and_pickle_free(tmp_path):
    manager = LITCheckpointManager(tmp_path, prefix="lit")
    expected = {"value": np.asarray([1.0, 2.0])}

    path = manager.save(7, expected)
    step, restored = manager.restore({"value": np.zeros(2)})

    assert step == 8
    np.testing.assert_array_equal(restored["value"], expected["value"])
    assert path.name == "lit_ckpt_000007.npz"
    assert not tuple(tmp_path.glob("*.tmp"))
    with np.load(path, allow_pickle=False) as archive:
        assert archive["step"].item() == 7


def test_lit_checkpoint_skips_a_corrupt_newest_file(tmp_path):
    manager = LITCheckpointManager(tmp_path, prefix="lit")
    fallback = {"value": np.zeros(1)}
    manager.save(2, {"value": np.asarray([2.0])})
    corrupt = tmp_path / "lit_ckpt_000003.npz"
    corrupt.write_bytes(b"not an npz archive")

    step, restored = manager.restore(fallback, strict=True)

    assert step == 3
    np.testing.assert_array_equal(restored["value"], [2.0])
