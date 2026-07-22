# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Primitive-correctness harness for Forward Laplacian handler tests.

These helpers parametrize over tracked input representations and compare
handler output with the dense chain-rule oracle in ``check_with_brute_force``.
Deterministic input seeding lives in ``tests.laplacian.input_fixtures``.
"""

import pytest

from tests.laplacian.helpers import check_with_brute_force
from tests.laplacian.input_fixtures import (
    InputCase,
    InputDomain,
    random_array,
    tracked_case_input,
)

TRACKED_INPUT_CASES: list[InputCase] = ["dense", "local1", "local2"]


def parametrize_over_tracked_cases(name):
    return pytest.mark.parametrize(
        name, [pytest.param(case) for case in TRACKED_INPUT_CASES]
    )


def parametrize_over_binary_cases(name):
    return pytest.mark.parametrize(
        name,
        [
            pytest.param(lhs, rhs)
            for lhs in TRACKED_INPUT_CASES
            for rhs in TRACKED_INPUT_CASES
        ]
        + [pytest.param(lhs, "plain") for lhs in TRACKED_INPUT_CASES]
        + [pytest.param("plain", rhs) for rhs in TRACKED_INPUT_CASES],
    )


def check_unary(
    fn,
    case: InputCase,
    *,
    domain: InputDomain = "real",
    shape: tuple[int, ...] = (3, 3),
    rtol=1e-5,
    atol=1e-10,
):
    """Check value/Jacobian/Laplacian of ``fn`` for one input case.

    Canonical usage, with the op table rows holding ``(fn, domain)``::

        @pytest.mark.parametrize(
            ("op", "domain"),
            (pytest.param(jnp.exp, "real", id="exp"), ...),
        )
        @parametrize_over_tracked_cases("case")
        def test_op(case, op, domain):
            check_unary(op, case, domain=domain)

    ``parametrize_over_tracked_cases`` intentionally supplies only pre-built
    ``LapTuple`` states. Raw-array auto-seeding is tested in
    ``tests/laplacian/public_api_test.py::TestInputSeeding``, not repeated for
    every unary primitive. ``shape`` covers tests that need a non-default
    coordinate layout, such as packed real coordinates for complex values.

    ``atol`` is floored to ``1e-6`` because the Hessian oracle is noisier than
    the default ``1e-10`` tolerance on some primitives.
    """
    x = random_array(domain, shape)
    check_with_brute_force(
        fn, tracked_case_input(x, case), rtol=rtol, atol=max(atol, 1e-6)
    )


def check_binary(
    fn,
    lhs: InputCase,
    rhs: InputCase,
    *,
    domain: InputDomain = "real",
    shape: tuple[int, ...] = (3, 3),
    rtol=1e-5,
    atol=1e-10,
):
    """Check ``fn(lhs, rhs)`` with independent tracked operand states.

    Both operands share the same tracked-basis shape (via ``input_shape`` in
    ``tracked_case_input``), but they use different primal values and Jacobian
    payloads so operand-swap and symmetry bugs cannot cancel invisibly. One-sided
    ``plain`` cases are kept here because a plain operand beside any ``LapTuple``
    is an untracked constant, which is a primitive-handler path rather than
    auto-seeding.

    ``atol`` is floored to ``5e-6`` because binary Hessian oracles are noisier
    than the unary default.
    """
    lhs_input = tracked_case_input(random_array(domain, shape, key=1), lhs, key=2)
    rhs_input = tracked_case_input(random_array(domain, shape, key=3), rhs, key=4)
    check_with_brute_force(fn, lhs_input, rhs_input, rtol=rtol, atol=max(atol, 5e-6))
