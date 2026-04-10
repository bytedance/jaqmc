# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Make kfac_jax be able to handle stochastic reconfiguration.

The KFAC algorithm itself is only applicable with real-valued outputs. But it
can be used to calculate the S matrix in stochastic reconfiguration method:

    S_{ij} = \langle
        \frac{\partial \log \psi^*}{\partial \theta_i}
        \frac{\partial \log \psi}{\partial \theta_j}
    \rangle

We only show the uncentered version for simplicity. After that, the parameters
($\theta$) will be updated following:

    \delta \theta = - Re[S]^{-1} Re[f]

where $f$ is the gradient of the loss function (total energy). The reason why we
are using $Re$ for $S$ and $f$ is that the parameter updates must be real. See
"Complex neural networks" paragraph of https://doi.org/10.1038/s41567-024-02566-1
[Nat. Phys. 20, 1476-1481 (2024)].

The above process is pretty much similar to the natural gradient decent (NGD).
In NGD, the gradients are pre-conditioned with the Fisher information matrix (FIM).
The first type of emperical FIM is defined as (following arXiv.2507.05127):

    \begin{aligned}
        F(\theta)=\frac{1}{N} \sum_n & (\mathrm{J}_{\theta} f_n)^{\top} \\
            & \mathbb{E}_{r(y \mid f_n)}[(-\nabla_{f_n} \log r(y \mid f_n)) \\
            & \quad(-\nabla_{f_n} \log r(y \mid f_n))^{\top}] \\
            & \mathrm{J}_{\theta} f_n
    \end{aligned}

After setting the entire $\mathbb{E}_{r(y \mid f_n)}[...]$ part as an constant and
setting $f_n$ as $\psi(x)$ (right) and $\psi^*(x)$ (left), the FIM will exactly become
the S matrix. It's easy to set the expectation part as an constant by using Gaussian
distribution (`register_normal_predictive_distribution` provided by kfac_jax). However,
setting $f_n$ as the complex wavefunction is a little bit tricky, as kfac_jax is written
for real-valued functions.

The key problem we need to solve is VJP for function with real inputs and complex
outputs. Reverse-mode autodiff (VJP) is used to calculate the Jacobian of outputs
$f$ with respect to the parameters of each layer. See `vjp_rc_with_aux` for details.

Setting the left one $f_n$ as $\psi^*(x)$ is handled in curvature_tags_and_blocks.py.
"""

from collections.abc import Callable
from typing import Any

import jax
from jax import numpy as jnp
from kfac_jax import BlockDiagonalCurvature
from kfac_jax._src.tracer import _layer_tag_vjp, cached_transformation

jax_vjp = jax.vjp  # capture original vjp for monkey patching


def tree_complex(re_tree, im_tree):
    return jax.tree.map(lambda re, im: re + 1j * im, re_tree, im_tree)


def vjp_rc_with_aux(fun: Callable, *primals, has_aux=True) -> tuple[Any, Callable, Any]:
    """Returns the VJP product for a function with real inputs and complex outputs.

    If you try `jax.vjp` with R->C functions and a complex vector, you will find JAX
    having real-valued outputs instead of complex-valued. This is understandable in the
    context of reverse-mode autodifferenciation, but is not what we want in mathematics.

    To fix this and get mathematically meaningful results, we need to split the real and
    imaginary part of the outputs and do VJP separately for them.
    """
    if not has_aux:
        raise NotImplementedError("kfac_jax only uses has_aux=True variant")

    def real_fun(*primals):
        val, aux = fun(*primals)
        return jax.tree.map(jnp.real, val), aux

    def imag_fun(*primals):
        val, aux = fun(*primals)
        return jax.tree.map(jnp.imag, val), aux

    vals_real, vjp_real_fun, aux = jax_vjp(real_fun, *primals, has_aux=True)
    if jnp.iscomplexobj(jax.tree.leaves(jax.eval_shape(fun, *primals))[0]):
        vals_imag, vjp_imag_fun, _ = jax_vjp(imag_fun, *primals, has_aux=True)

        def vjp_fun_rc(y):
            return tree_complex(vjp_real_fun(y), vjp_imag_fun(y))  # y is always real

        return tree_complex(vals_real, vals_imag), vjp_fun_rc, aux
    else:
        return vals_real, vjp_real_fun, aux


def patched_layer_tag_vjp(processed_jaxpr, primal_func_args):
    """Computes primal values and tangents w.r.t. all layer tags.

    Monkey patching the original version with the one supporting R->C.

    Args:
        processed_jaxpr: The :class:`~ProcessedJaxpr` representing the model
            function. This must include at least one loss tag.
        primal_func_args: The primals at which to evaluate the Hessian.

    Returns:
        The computed ``losses`` and ``vjp_func`` pair.
    """
    jax.vjp = vjp_rc_with_aux
    try:
        return _layer_tag_vjp(processed_jaxpr, primal_func_args)
    finally:
        jax.vjp = jax_vjp


def patch_block_diagonal_curvature(estimator: BlockDiagonalCurvature):
    # use VJP transformation from our patched version
    estimator._vjp, estimator._jaxpr_extractor = cached_transformation(
        func=estimator.func,  # type: ignore
        transformation=patched_layer_tag_vjp,
        params_index=0,
        auto_register_tags=True,
        allow_left_out_params=False,
        raise_error_on_diff_jaxpr=True,
        **estimator._auto_register_kwargs,
    )


class PatchedBlockDiagonalCurvature(BlockDiagonalCurvature):
    """Block diagonal curvature estimator class with R->C VJP support."""

    def __init__(self, func, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        patch_block_diagonal_curvature(self)
