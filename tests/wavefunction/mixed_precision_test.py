# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Precision boundaries, strategies, and evaluation entrypoints for wavefunctions."""

from collections.abc import Callable
from typing import Any, cast

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.wavefunction._dressing import DressingConfig
from jaqmc.app.hall.wavefunction.mhpo import MHPO
from jaqmc.app.hydrogen_atom import HydrogenAtom, HydrogenAtomData
from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.wavefunction.ferminet import FermiNetWavefunction
from jaqmc.app.molecule.wavefunction.lapnet import LapNetWavefunction
from jaqmc.app.molecule.wavefunction.psiformer import PsiformerWavefunction
from jaqmc.app.solid.data import SolidData
from jaqmc.app.solid.wavefunction import SolidWavefunction
from jaqmc.geometry.sphere import (
    spinor_coordinates_from_stereographic,
    stereographic_coordinates,
)
from jaqmc.laplacian import forward_laplacian, make_laplacian_input
from jaqmc.laplacian.precision import ContractionMode, jacobian_contraction_scope
from jaqmc.utils.config import ConfigManager
from jaqmc.wavefunction.backbone.lapnet import lapnet_sparse_attention
from jaqmc.wavefunction.backbone.layernorm import LayerNorm
from jaqmc.wavefunction.output.logdet import LogDet
from jaqmc.wavefunction.output.orbital import OrbitalProjection
from tests.laplacian.helpers import brute_force_oracle


class TestComponentPrecisionBoundaries:
    """Precision casting and promotion rules in reusable wavefunction layers."""

    def test_layernorm_param_compute_dtype_rounds_parameters(self) -> None:
        """LayerNorm rounds scale and bias to param_compute_dtype before applying."""
        x = jnp.linspace(-0.5, 0.5, 6, dtype=jnp.float32).reshape(3, 2)
        norm_params = {
            "params": {
                "scale": jnp.array([0.7013, 1.3017], dtype=jnp.float32),
                "bias": jnp.array([-0.2013, 0.3017], dtype=jnp.float32),
            }
        }
        rounded_params = jax.tree.map(lambda v: v.astype(jnp.float16), norm_params)
        out_fp16 = cast(
            jnp.ndarray,
            LayerNorm(dtype=jnp.float32, param_compute_dtype=jnp.float16).apply(
                norm_params, x
            ),
        )
        out_rounded = cast(
            jnp.ndarray,
            LayerNorm(dtype=jnp.float32, param_compute_dtype=jnp.float32).apply(
                rounded_params, x
            ),
        )
        assert out_fp16.dtype == jnp.float32
        np.testing.assert_array_equal(out_fp16, out_rounded)

    @pytest.mark.requires_x64
    def test_orbital_projection_promotes_fp16_output_to_fp32(self) -> None:
        """OrbitalProjection computes in dtype but promotes FP16 outputs to FP32."""
        x = jnp.linspace(-0.5, 0.5, 6, dtype=jnp.float32).reshape(3, 2)
        for split in (False, True):
            proj = OrbitalProjection(
                (2, 1), ndets=2, orbitals_spin_split=split, dtype=jnp.float16
            )
            orbitals = cast(
                jnp.ndarray,
                proj.apply(proj.init(jax.random.key(0), x), x.astype(jnp.float16)),
            )
            assert orbitals.dtype == jnp.float32
            assert orbitals.shape == (2, 3, 3)

        proj_f64 = OrbitalProjection((2, 1), ndets=2, dtype=None)
        x_f64 = x.astype(jnp.float64)
        out_f64 = cast(
            jnp.ndarray,
            proj_f64.apply(proj_f64.init(jax.random.key(1), x_f64), x_f64),
        )
        assert out_f64.dtype == jnp.float64

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        ("input_dtype", "cap", "expected_dtype"),
        [
            (jnp.float16, None, jnp.float32),
            (jnp.float64, None, jnp.float64),
            (jnp.complex128, None, jnp.complex128),
            (jnp.complex128, jnp.float32, jnp.complex64),
        ],
    )
    def test_logdet_promotes_fp16_and_caps_precision(
        self, input_dtype: type, cap: type | None, expected_dtype: type
    ) -> None:
        """LogDet promotes FP16 to FP32 and caps higher-precision inputs."""
        matrices = jnp.array([[[2, 0], [0, 3]], [[-1, 0], [0, 2]]], dtype=input_dtype)
        if jnp.issubdtype(input_dtype, jnp.complexfloating):
            matrices = matrices.at[0, 0, 0].set(2j)
        actual = cast(dict[str, jnp.ndarray], LogDet(dtype=cap).apply({}, matrices))
        expected_signs, expected_logs = np.linalg.slogdet(
            np.asarray(matrices, expected_dtype)
        )
        assert actual["logpsi"].dtype == expected_dtype
        assert actual["abs_logdets"].dtype == jnp.empty((), expected_dtype).real.dtype
        np.testing.assert_allclose(actual["sign_logdets"], expected_signs, rtol=1e-6)
        np.testing.assert_allclose(actual["abs_logdets"], expected_logs, rtol=1e-6)

    @pytest.mark.parametrize(
        ("dtype", "mode", "sparse_axis", "tolerance"),
        [
            (jnp.float32, ContractionMode.STANDARD, 0, 2e-5),
            (jnp.float32, ContractionMode.SCALED_FP16, 0, 3e-3),
            (jnp.float16, ContractionMode.SCALED_FP16, 0, 3e-3),
            (jnp.float16, ContractionMode.SCALED_FP16, None, 3e-3),
        ],
    )
    def test_lapnet_attention_forward_laplacian(
        self,
        dtype: type,
        mode: ContractionMode,
        sparse_axis: int | None,
        tolerance: float,
    ) -> None:
        """Local1 and dense attention preserve dtypes and native derivatives."""
        features = jax.random.normal(jax.random.key(21), (3, 2, 2), dtype)
        value = jax.random.normal(jax.random.key(9), features.shape, dtype)

        def evaluate(x: jnp.ndarray) -> jnp.ndarray:
            with jacobian_contraction_scope(mode):
                attention = lapnet_sparse_attention(
                    jnp.tanh(x),
                    0.4 * x + 0.2,
                    jnp.sin(x) if sparse_axis == 0 else value,
                )
                return attention + x

        actual = jax.jit(forward_laplacian(evaluate))(
            make_laplacian_input(features, sparse_axis=sparse_axis)
        )
        reference = jax.jit(lambda x: brute_force_oracle(evaluate, x))(features)
        assert (
            actual.x.dtype
            == actual.dense_jacobian.dtype
            == actual.laplacian.dtype
            == dtype
        )
        np.testing.assert_allclose(actual.x, reference.x, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            actual.dense_jacobian,
            reference.dense_jacobian,
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            actual.laplacian, reference.laplacian, rtol=tolerance, atol=tolerance
        )


class TestWavefunctionPrecisionContract:
    """Configuration, runtime overrides, and parameter sharing across modes."""

    def test_base_wavefunction_rejects_explicit_precision_mode(self) -> None:
        """Wavefunctions without configurable precision reject non-None precision."""
        h_wf = HydrogenAtom()
        h_data = HydrogenAtomData(jnp.array([[1.0, 2.0, 2.0]]))
        h_params = h_wf.init_params(h_data, jax.random.key(0))
        np.testing.assert_allclose(
            h_wf.evaluate(h_params, h_data, precision=None), -2.4
        )
        with pytest.raises(ValueError, match="does not support precision mode"):
            h_wf.evaluate(h_params, h_data, precision="fp16_mixed")

    def test_config_manager_resolves_precision_override(self) -> None:
        """ConfigManager resolves wf.precision CLI overrides on wavefunction modules."""
        cfg = ConfigManager(
            {"wf": {"module": LapNetWavefunction.__module__}},
            ["wf.precision=fp16_mixed"],
        )
        assert cfg.get_module("wf").precision == "fp16_mixed"

    def test_runtime_precision_override_clones_subclass_with_shared_params(
        self,
    ) -> None:
        """Runtime precision overrides clone the module and preserve subclass hooks."""

        class ScaledOrbitalLapNet(LapNetWavefunction):
            def get_orbitals(self, data: MoleculeData) -> jnp.ndarray:
                return 2 * super().get_orbitals(data)

        kw: dict[str, Any] = dict(
            nspins=(1, 1), ndets=1, num_layers=1, num_heads=2, heads_dim=4
        )
        wf = ScaledOrbitalLapNet(**kw, precision="fp16_jac")
        base = LapNetWavefunction(**kw, precision="fp16_mixed")
        data = MoleculeData(
            jax.random.normal(jax.random.key(7), (2, 3)),
            jnp.zeros((1, 3)),
            jnp.array([2.0]),
        )
        params = wf.init_params(data, jax.random.key(0))

        np.testing.assert_array_equal(
            wf.logpsi(params, data), wf.logpsi(params, data, precision="fp16_jac")
        )
        np.testing.assert_allclose(
            jax.jit(wf.logpsi, static_argnames=("precision",))(
                params, data, precision="fp16_mixed"
            ),
            jax.jit(base.logpsi)(params, data) + 2 * jnp.log(2.0),
            rtol=1e-6,
        )
        assert wf.orbitals(params, data, precision="fp16_mixed").shape == (1, 2, 2)

    @pytest.mark.requires_x64
    def test_fp16_modes_cap_output_dtype_without_mutating_master_params(self) -> None:
        """FP16 modes cap outputs to FP32 while keeping master parameters untouched."""
        wf = LapNetWavefunction(
            nspins=(1, 1),
            ndets=1,
            num_layers=1,
            num_heads=2,
            heads_dim=4,
            precision="fp16_jac",
        )
        data = MoleculeData(
            jax.random.normal(jax.random.key(7), (2, 3), dtype=jnp.float64),
            jnp.zeros((1, 3), dtype=jnp.float64),
            jnp.array([2.0], dtype=jnp.float64),
        )
        params = jax.tree.map(
            lambda v: v.astype(jnp.float64), wf.init_params(data, jax.random.key(0))
        )
        original = jax.tree.map(np.copy, params)

        for precision, expected_dtype in (
            ("standard", jnp.float64),
            ("fp16_jac", jnp.float32),
            ("fp16_mixed", jnp.float32),
        ):
            output = wf.evaluate(params, data, precision=precision)
            assert output["abs_logdets"].dtype == expected_dtype

        for before, after in zip(
            jax.tree.leaves(original), jax.tree.leaves(params), strict=True
        ):
            assert after.dtype == jnp.float64
            np.testing.assert_array_equal(after, before)


def _assert_precision_primal_and_derivatives(
    eval_fn: Callable[..., jnp.ndarray],
    params: Any,
    electrons: jnp.ndarray,
    make_input: Callable[[Any], Any],
    *,
    lap_atol: float = 1e-2,
) -> None:
    """Verify primal values and derivatives for fp16_jac and fp16_mixed."""
    primal_input = make_input(electrons)
    seeded = make_input(make_laplacian_input(electrons, sparse_axis=0))

    standard_primal = jax.jit(lambda p, x: eval_fn(p, x, precision="standard"))(
        params, primal_input
    )
    standard = jax.jit(forward_laplacian(eval_fn, precision="standard"))(params, seeded)
    np.testing.assert_allclose(standard.x, standard_primal, rtol=1e-6, atol=1e-6)

    # 1. fp16_jac: FP32 forward pass + scaled FP16 Jacobian contractions.
    # Primal matches standard FP32; derivatives use scaled FP16 contractions.
    jac_primal = jax.jit(lambda p, x: eval_fn(p, x, precision="fp16_jac"))(
        params, primal_input
    )
    jac_lap = jax.jit(forward_laplacian(eval_fn, precision="fp16_jac"))(params, seeded)
    assert jac_lap.x.dtype == standard.x.dtype
    np.testing.assert_allclose(jac_primal, standard_primal, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jac_lap.x, jac_primal, rtol=1e-6, atol=1e-6)
    assert not np.array_equal(jac_lap.dense_jacobian, standard.dense_jacobian)
    assert not np.array_equal(jac_lap.laplacian, standard.laplacian)
    np.testing.assert_allclose(
        jac_lap.dense_jacobian, standard.dense_jacobian, rtol=3e-3, atol=3e-3
    )
    np.testing.assert_allclose(
        jac_lap.laplacian, standard.laplacian, rtol=3e-3, atol=lap_atol
    )

    # 2. fp16_mixed: FP16 backbone/orbital projections + scaled FP16 contractions.
    # Primal and derivatives match the brute-force oracle of the fp16_mixed function
    # tightly, and remain close to the standard FP32 evaluation.
    mixed_primal = jax.jit(lambda p, x: eval_fn(p, x, precision="fp16_mixed"))(
        params, primal_input
    )
    mixed_lap = jax.jit(forward_laplacian(eval_fn, precision="fp16_mixed"))(
        params, seeded
    )
    mixed_oracle = jax.jit(
        lambda p, x: brute_force_oracle(
            lambda pos: eval_fn(p, make_input(pos), precision="fp16_mixed"),
            x,
        )
    )(params, electrons)
    assert mixed_lap.x.dtype == standard.x.dtype
    assert not np.array_equal(mixed_primal, standard_primal)
    assert not np.array_equal(mixed_lap.dense_jacobian, standard.dense_jacobian)
    assert not np.array_equal(mixed_lap.laplacian, standard.laplacian)
    np.testing.assert_allclose(mixed_lap.x, mixed_primal, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mixed_lap.x, mixed_oracle.x, rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(mixed_primal, standard_primal, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(
        mixed_lap.dense_jacobian, mixed_oracle.dense_jacobian, rtol=3e-3, atol=3e-3
    )
    np.testing.assert_allclose(
        mixed_lap.laplacian, mixed_oracle.laplacian, rtol=3e-3, atol=lap_atol
    )
    np.testing.assert_allclose(
        mixed_lap.dense_jacobian, standard.dense_jacobian, rtol=2e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        mixed_lap.laplacian, standard.laplacian, rtol=3e-2, atol=2e-2
    )


class TestWavefunctionPrecisionEvaluation:
    """Primal and Forward Laplacian consistency across precision modes."""

    @pytest.mark.parametrize(
        "wf",
        [
            FermiNetWavefunction(
                nspins=(1, 1),
                ndets=1,
                hidden_dims_single=[8],
                hidden_dims_double=[4],
                precision="fp16_jac",
            ),
            LapNetWavefunction(
                nspins=(1, 1),
                ndets=1,
                num_layers=1,
                num_heads=2,
                heads_dim=4,
                use_layernorm=True,
                precision="fp16_jac",
            ),
            PsiformerWavefunction(
                nspins=(1, 1),
                ndets=1,
                num_layers=1,
                num_heads=2,
                heads_dim=4,
                mlp_hidden_dims=[8],
                precision="fp16_jac",
            ),
        ],
        ids=["ferminet", "lapnet", "psiformer"],
    )
    def test_molecule_wavefunctions(
        self,
        wf: FermiNetWavefunction | LapNetWavefunction | PsiformerWavefunction,
    ) -> None:
        data = MoleculeData(
            jax.random.normal(jax.random.key(7), (2, 3)),
            jnp.zeros((1, 3)),
            jnp.array([2.0]),
        )
        params = wf.init_params(data, jax.random.key(0))
        _assert_precision_primal_and_derivatives(
            wf.logpsi,
            params,
            data.electrons,
            lambda x: data.merge({"electrons": x}),
            lap_atol=5e-3,
        )

    def test_solid_wavefunction(self) -> None:
        solid_wf = SolidWavefunction(
            nspins=(1, 1),
            ndets=1,
            klist=jnp.array([[0.0, 0.0, 0.0], [0.25, -0.15, 0.1]]),
            simulation_lattice=jnp.eye(3, dtype=jnp.float32) * 8.0,
            primitive_lattice=jnp.eye(3, dtype=jnp.float32) * 4.0,
            hidden_dims_single=[8],
            hidden_dims_double=[4],
            precision="fp16_jac",
        )
        solid_data = SolidData(
            electrons=jax.random.normal(jax.random.key(7), (2, 3), dtype=jnp.float32),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            primitive_atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.array([2.0], dtype=jnp.float32),
        )
        params = solid_wf.init_params(solid_data, jax.random.key(0))
        assert solid_wf.orbitals(params, solid_data, precision="fp16_mixed").shape == (
            1,
            2,
            2,
        )
        _assert_precision_primal_and_derivatives(
            solid_wf.logpsi,
            params,
            solid_data.electrons,
            lambda x: solid_data.merge({"electrons": x}),
        )

    def test_mhpo_wavefunction(self) -> None:
        mhpo_wf = MHPO(
            nspins=(1, 1),
            monopole_strength=1.0,
            flux=2,
            interaction_strength=1.0,
            radius=1.0,
            ndets=1,
            backbone="psiformer",
            num_heads=2,
            heads_dim=4,
            num_layers=1,
            orbital_readout="factorized",
            dressing=DressingConfig(type="full", degree=2, random_init=False),
            precision="fp16_jac",
        )
        electrons = jnp.array([[0.7, 0.2], [1.4, -0.9]], dtype=jnp.float32)
        hall_data = HallData(electrons=electrons)
        mhpo_params = mhpo_wf.init_params(hall_data, jax.random.key(0))
        np.testing.assert_allclose(
            mhpo_wf.logpsi(mhpo_params, hall_data, precision="fp16_mixed"),
            mhpo_wf.logpsi(mhpo_params, hall_data, precision="standard"),
            rtol=5e-3,
            atol=5e-3,
        )
        plane, is_north = stereographic_coordinates(electrons)

        def spinor_eval(p: Any, x: jnp.ndarray, *, precision: str | None = None):
            u, v = spinor_coordinates_from_stereographic(x, is_north)
            return mhpo_wf.logpsi_from_spinor(p, u, v, precision=precision)

        _assert_precision_primal_and_derivatives(
            spinor_eval, mhpo_params, plane, lambda x: x, lap_atol=0.1
        )
