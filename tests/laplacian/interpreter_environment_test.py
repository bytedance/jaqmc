# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JaxExprEnvironment and interpreter helper functions."""

import operator
from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import JaxprEqn

from jaqmc.laplacian.interpreter import (
    JaxExprEnvironment,
    _eval_eqn,
    _resolve_bind_params,
)


def _walk_graph(env, jaxpr):
    """Execute all equations through env, return list of freed vars per step."""
    freed_per_step = []
    for eqn in jaxpr.eqns:
        before = set(env.env.keys())
        invals = env.read_many(eqn.invars)
        after_read = set(env.env.keys())
        freed_per_step.append(before - after_read)

        outval = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outval = [outval]
        env.write_many(eqn.outvars, outval)
    return freed_per_step


class _FakePrimitive:
    def __init__(self):
        self.multiple_results = False
        self.last_call = None

    def get_bind_params(self, params):
        del params
        return {
            "scale": 3,
            "subfuns": (lambda x: x + 1.0, lambda x: x * 2.0),
        }

    def bind(self, *args, **kwargs):
        self.last_call = (args, kwargs)
        return args, kwargs


class _OldStylePrimitive(_FakePrimitive):
    def get_bind_params(self, params):
        del params
        return (
            (lambda x: x + 1.0, lambda x: x * 2.0),
            {"scale": 3},
        )


class TestBindParams:
    """Verify interpreter bind-param normalization and dispatch."""

    def test_plain_primitive_has_no_bind_subfuns(self):
        closed = jax.make_jaxpr(lambda x: x + 1.0)(1.0)
        eqn = closed.jaxpr.eqns[0]

        old_style, subfuns, params = _resolve_bind_params(eqn)

        assert isinstance(old_style, bool)
        assert subfuns == ()
        assert params == {}

    def test_custom_jvp_call_reports_subfuns_separately(self):
        @jax.custom_jvp
        def fn(x):
            return jnp.sin(x)

        @fn.defjvp
        def fn_jvp(primals, tangents):
            (x,) = primals
            (x_dot,) = tangents
            return fn(x), 2 * x_dot * jnp.cos(x)

        closed = jax.make_jaxpr(fn)(1.0)
        eqn = closed.jaxpr.eqns[0]

        old_style, subfuns, params = _resolve_bind_params(eqn)

        assert eqn.primitive.name == "custom_jvp_call"
        assert isinstance(old_style, bool)
        assert params["symbolic_zeros"] is False
        assert len(subfuns) == 2

    def test_modern_bind_contract_extracts_subfuns(self):
        primitive = _FakePrimitive()
        eqn = cast(JaxprEqn, SimpleNamespace(primitive=primitive, params={}))

        old_style, subfuns, params = _resolve_bind_params(eqn)

        assert old_style is False
        assert params["scale"] == 3
        assert len(subfuns) == 2

    def test_modern_bind_dispatch_passes_subfuns_as_kwargs(self):
        primitive = _FakePrimitive()
        eqn = cast(JaxprEqn, SimpleNamespace(primitive=primitive, params={}))

        result = _eval_eqn(eqn, (jnp.asarray(4.0),))

        assert primitive.last_call is not None
        args, kwargs = primitive.last_call
        assert args == (4.0,)
        assert kwargs["scale"] == 3
        assert len(kwargs["subfuns"]) == 2
        assert result == primitive.last_call

    def test_old_style_bind_dispatch_passes_subfuns_positionally(self):
        primitive = _OldStylePrimitive()
        eqn = cast(JaxprEqn, SimpleNamespace(primitive=primitive, params={}))

        old_style, subfuns, params = _resolve_bind_params(eqn)

        assert old_style is True
        assert params["scale"] == 3
        assert len(subfuns) == 2

        result = _eval_eqn(eqn, (jnp.asarray(5.0),))

        assert primitive.last_call is not None
        args, kwargs = primitive.last_call
        assert len(args) == 3
        assert callable(args[0])
        assert callable(args[1])
        np.testing.assert_allclose(args[2], 5.0)
        assert kwargs["scale"] == 3
        assert result == primitive.last_call


class TestMemoryFreeing:
    """Verify that variables are freed from env after their last read."""

    def test_multi_use_intermediate_survives_until_final_read(self):
        """A reused intermediate is available for both consumers, then freed."""
        closed = jax.make_jaxpr(lambda x: (y := jax.numpy.sin(x)) + y)(1.0)
        jaxpr = closed.jaxpr
        env = JaxExprEnvironment(jaxpr, closed.consts, jnp.array(1.0))

        sin_outvar = jaxpr.eqns[0].outvars[0]
        assert env.reference_counter[sin_outvar] == 2

        invals = env.read_many(jaxpr.eqns[0].invars)
        env.write_many(
            jaxpr.eqns[0].outvars,
            [jaxpr.eqns[0].primitive.bind(*invals, **jaxpr.eqns[0].params)],
        )
        assert sin_outvar in env.env

        first_read = env.read(sin_outvar)
        assert sin_outvar in env.env
        assert env.reference_counter[sin_outvar] == 1

        second_read = env.read(sin_outvar)
        assert sin_outvar not in env.env
        assert sin_outvar not in env.reference_counter
        np.testing.assert_allclose(first_read, second_read)

    def test_read_many_returns_correct_values(self):
        """read_many returns correct values even as it frees variables."""
        closed = jax.make_jaxpr(operator.add)(1.0, 2.0)
        jaxpr = closed.jaxpr
        env = JaxExprEnvironment(jaxpr, closed.consts, jnp.array(1.0), jnp.array(2.0))

        eqn = jaxpr.eqns[0]
        vals = env.read_many(eqn.invars)
        # Values should be returned before freeing
        assert int(np.asarray(vals[0])) == 1
        assert int(np.asarray(vals[1])) == 2


class TestChainedGraph:
    """Test a longer computation graph with multiple intermediates."""

    def test_chain_frees_all_intermediates(self):
        """In sin(cos(exp(x))), all intermediates except output are freed."""
        closed = jax.make_jaxpr(
            lambda x: jax.numpy.sin(jax.numpy.cos(jax.numpy.exp(x)))
        )(1.0)
        jaxpr = closed.jaxpr
        env = JaxExprEnvironment(jaxpr, closed.consts, jnp.array(1.0))

        _walk_graph(env, jaxpr)

        # Only output vars and invars (with write ref) should remain
        for v in env.env:
            assert v in jaxpr.outvars or v in jaxpr.invars

    def test_final_read_many_returns_outputs(self):
        """The full eval pattern: walk graph then read_many(outvars)."""
        closed = jax.make_jaxpr(lambda x: jax.numpy.sin(jax.numpy.cos(x)))(1.0)
        jaxpr = closed.jaxpr
        env = JaxExprEnvironment(jaxpr, closed.consts, jnp.array(1.0))

        _walk_graph(env, jaxpr)
        results = env.read_many(jaxpr.outvars)

        expected = float(jax.numpy.sin(jax.numpy.cos(1.0)))
        assert abs(float(np.asarray(results[0])) - expected) < 1e-10

    def test_dropped_output_is_not_stored(self):
        """JAX DropVar outvars are ignored because they have no references."""
        closed = jax.make_jaxpr(lambda x: jax.lax.top_k(x, 2)[0])(jnp.arange(4.0))
        jaxpr = closed.jaxpr
        (top_k_eqn,) = jaxpr.eqns
        env = JaxExprEnvironment(jaxpr, closed.consts, jnp.arange(4.0))

        kept_outvar, dropped_outvar = top_k_eqn.outvars
        assert type(dropped_outvar).__name__ == "DropVar" or str(dropped_outvar) == "_"
        assert dropped_outvar not in env.reference_counter

        invals = env.read_many(top_k_eqn.invars)
        outvals = top_k_eqn.primitive.bind(*invals, **top_k_eqn.params)
        env.write_many(top_k_eqn.outvars, outvals)

        assert kept_outvar in env.env
        assert dropped_outvar not in env.env
