# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.utils.checkpoint import NumPyCheckpointManager, tree_from_npz, tree_to_npz


def _fallback_tree():
    return {
        "scalar": 0,
        "np": np.array([[0.0, 0.0]]),
        "nested": {"jax": jnp.array([0, 0])},
    }


def _restored_tree():
    return {
        "scalar": 7,
        "np": np.array([[1.5, 2.5]]),
        "nested": {"jax": jnp.array([3, 4])},
    }


def _corrupt_file(path):
    path.write_bytes(path.read_bytes()[:-10])


def test_save_and_restore_round_trip_with_prefix(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="train")
    fallback = _fallback_tree()
    expected = _restored_tree()

    manager.save(4, expected)

    ckpt_path = tmp_path / "train_ckpt_000004.npz"
    assert ckpt_path.exists()

    step, restored = manager.restore(fallback)

    assert step == 5
    assert restored["scalar"] == expected["scalar"]
    np.testing.assert_array_equal(restored["np"], expected["np"])
    np.testing.assert_array_equal(
        np.asarray(restored["nested"]["jax"]),
        np.asarray(expected["nested"]["jax"]),
    )


def test_restore_uses_restore_path_file(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="train")
    expected = _restored_tree()
    manager.save(2, expected)

    ckpt_path = tmp_path / "train_ckpt_000002.npz"
    direct_file_manager = NumPyCheckpointManager(tmp_path, ckpt_path, prefix="ignored")

    step, restored = direct_file_manager.restore(_fallback_tree())

    assert step == 3
    assert restored["scalar"] == expected["scalar"]
    np.testing.assert_array_equal(restored["np"], expected["np"])


def test_restore_from_file_rejects_non_file_path(tmp_path):
    with pytest.raises(ValueError, match="is not a file"):
        NumPyCheckpointManager.restore_from_file(tmp_path, _fallback_tree())


def test_restore_returns_fallback_when_path_missing(tmp_path):
    manager = NumPyCheckpointManager(tmp_path / "missing", prefix="train")
    fallback = _fallback_tree()

    step, restored = manager.restore(fallback)

    assert step == 0
    assert restored is fallback


def test_restore_raises_when_path_missing_in_strict_mode(tmp_path):
    manager = NumPyCheckpointManager(tmp_path / "missing", prefix="train")

    with pytest.raises(FileNotFoundError, match="Checkpoint path does not exist"):
        manager.restore(_fallback_tree(), strict=True)


def test_restore_returns_fallback_when_no_matching_checkpoint_files(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="train")
    fallback = _fallback_tree()
    (tmp_path / "other_ckpt_000001.npz").write_bytes(b"placeholder")

    step, restored = manager.restore(fallback)

    assert step == 0
    assert restored is fallback


def test_restore_raises_when_no_matching_checkpoint_files_in_strict_mode(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="train")

    with pytest.raises(FileNotFoundError, match="No matching checkpoints found"):
        manager.restore(_fallback_tree(), strict=True)


def test_restore_skips_bad_latest_checkpoint(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="train")
    expected = _restored_tree()

    manager.save(2, expected)
    manager.save(4, _fallback_tree())
    _corrupt_file(tmp_path / "train_ckpt_000004.npz")

    step, restored = manager.restore(_fallback_tree())

    assert step == 3
    assert restored["scalar"] == expected["scalar"]
    np.testing.assert_array_equal(restored["np"], expected["np"])


def test_restore_skips_bad_latest_checkpoint_in_strict_mode(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="train")
    expected = _restored_tree()

    manager.save(2, expected)
    manager.save(4, _fallback_tree())
    _corrupt_file(tmp_path / "train_ckpt_000004.npz")

    step, restored = manager.restore(_fallback_tree(), strict=True)

    assert step == 3
    assert restored["scalar"] == expected["scalar"]
    np.testing.assert_array_equal(restored["np"], expected["np"])


def test_restore_returns_fallback_when_all_matching_checkpoints_are_bad(tmp_path):
    manager = NumPyCheckpointManager(tmp_path, prefix="train")
    fallback = _fallback_tree()

    manager.save(2, _restored_tree())
    manager.save(4, _fallback_tree())
    _corrupt_file(tmp_path / "train_ckpt_000002.npz")
    _corrupt_file(tmp_path / "train_ckpt_000004.npz")

    step, restored = manager.restore(fallback)

    assert step == 0
    assert restored is fallback


def test_restore_raises_when_all_matching_checkpoints_are_bad_in_strict_mode(
    tmp_path,
):
    manager = NumPyCheckpointManager(tmp_path, prefix="train")

    manager.save(2, _restored_tree())
    manager.save(4, _fallback_tree())
    _corrupt_file(tmp_path / "train_ckpt_000002.npz")
    _corrupt_file(tmp_path / "train_ckpt_000004.npz")

    with pytest.raises(RuntimeError, match="Failed to restore any checkpoint"):
        manager.restore(_fallback_tree(), strict=True)


def test_tree_to_npz_and_tree_from_npz_round_trip():
    tree = _restored_tree()
    fallback = _fallback_tree()

    npz_data = tree_to_npz(tree)

    assert {"scalar", "np", "nested/jax"} <= set(npz_data)

    restored = tree_from_npz(npz_data, fallback)

    assert restored["scalar"] == tree["scalar"]
    np.testing.assert_array_equal(restored["np"], tree["np"])
    np.testing.assert_array_equal(
        np.asarray(restored["nested"]["jax"]),
        np.asarray(tree["nested"]["jax"]),
    )
