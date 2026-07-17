# Forward Laplacian Internals

This page is for contributors who need to understand how Forward Laplacian
executes and where to extend it. Use <project:custom-rules.md> when one helper
function needs a handwritten `@custom_laplacian` rule. Stay on this page when
the problem is lower-level: primitive dispatch, interpreter behavior, or sparse
retention versus dense fallback.

Most readers arrive here with one of three jobs:

- explain why a particular equation took the dense path
- decide whether a primitive needs a wrapper or a dedicated handler
- preserve sparse Jacobian structure across a primitive or staged-call boundary

Before reading it, make sure you are already comfortable with:

- the public Forward Laplacian API in <project:index.md>
- common JAX concepts: transforms, pytrees, `jaxpr`, and primitives.
  Read <project:../jax-for-jaqmc.md> for essential background
- the difference between helper-level and interpreter-level customization:
  helper-level customization changes one function, while interpreter-level
  customization changes how Forward Laplacian handles a primitive or staged call

On this page, "tracked" means "this value is currently carrying derivative
state." A tracked value is a
{class}`~jaqmc.laplacian.LapTuple` containing a primal value, a Jacobian
payload, and an accumulated Laplacian.

The package is organized around four reader-facing layers:

- state model
- tracing and execution engine
- helper-level custom-rule authoring
- primitive plumbing and primitive rules

## State Model

Forward Laplacian propagates one logical object through the computation. For
each tracked intermediate value, JaQMC keeps

- the primal output
- a Jacobian with respect to the tracked scalar inputs
- the propagated Laplacian

Those three fields travel together in
{class}`~jaqmc.laplacian.LapTuple`.

The main internal distinction is how the Jacobian is stored.

- Dense path: `LapTuple.jacobian` is an array with the tracked-input basis on
  the leading axis.
- Sparse path: `LapTuple.jacobian` is a structured payload that represents the
  same Jacobian in an owner-local form.

At public boundaries, think in terms of the dense contract
`(n, *x.shape)`. Use
{attr}`~jaqmc.laplacian.LapTuple.dense_jacobian` whenever you need that uniform
view. The storage form only matters when you are trying to preserve sparsity
inside primitive handling.

The derivative basis stays leading deliberately. Each dense Jacobian slice then
has the same layout as the primal value, so dense dot and matrix operations can
contract the real trailing matrix axes through the usual GPU GEMM-friendly
layout. Moving the derivative basis to the final axis makes that artificial
basis axis look like the last contraction dimension and tends to force less
favorable transposes or strided contractions in core dense paths.

## Tracked and Untracked Inputs

The tracked/untracked boundary is fixed before the interpreter starts walking
the `jaxpr`.

If every input leaf is a plain array, Forward Laplacian seeds every leaf with a
default tracked state:

- the primal value is the original leaf
- the Jacobian starts as an identity basis over all tracked scalar inputs
- the Laplacian starts at zero

If any input leaf is already a `LapTuple`, JaQMC keeps those tracked leaves
as-is and treats every remaining plain array leaf as an untracked constant.
That all-or-nothing rule is what makes
{func}`~jaqmc.laplacian.make_laplacian_input` behave like "track only these
inputs." It does not partially reseed the rest of the tree for you.

Sparse tracking also starts from
{func}`~jaqmc.laplacian.make_laplacian_input`, but with `sparse_axis=...`.
Instead of building a dense identity Jacobian, JaQMC seeds an owner-local
sparse Jacobian so downstream primitives can try to preserve that structure.

In practice, a primitive handler may therefore see:

- plain arrays, meaning "not tracked"
- dense `LapTuple` values
- sparse `LapTuple` values seeded from an owner-local input
- mixtures of tracked and untracked inputs in the same primitive call

## Execution Model

At a high level, Forward Laplacian evaluation does this:

1. trace `fn` to a `jaxpr` using primal array values only
2. build the initial tracked state for the chosen inputs
3. walk the `jaxpr` equation by equation
4. either bind the original primitive directly or dispatch to
   Laplacian-aware handling
5. rebuild the original output PyTree from the per-equation results

The important split is that tracing and derivative propagation happen in
separate phases. The trace sees only primal arrays. The later interpreter walk
sees a mixture of plain arrays and `LapTuple` values and propagates derivative
state explicitly.

While it walks the `jaxpr`, the interpreter keeps an environment for variables,
stores inputs, constants, and intermediates by variable, and releases
intermediates after their final downstream use. That keeps the execution model
close to the `jaxpr` itself without retaining every value for the full graph.

If you need to inspect the object the interpreter is walking, this is the
relevant shape:

```python
import jax


jaxpr = jax.make_jaxpr(fn)(*example_args)
for eqn in jaxpr.jaxpr.eqns:
    print(eqn.primitive.name, eqn.params)
```

## Dispatch and Extension Boundaries

Once the interpreter is walking equations, the first question is simple: does
this equation have any tracked inputs?

- If no, JaQMC binds the original primitive directly.
- If yes, JaQMC dispatches to Forward Laplacian logic.

Most tracked equations go through the primitive registry in
{mod}`jaqmc.laplacian.primitives`. Built-in handlers are grouped across modules
such as arithmetic, elementwise, selection/indexing, miscellaneous wrapper-led
rules, structure, and linear algebra rules.

If a primitive has no registered handler, JaQMC falls back to a generic dense
rule. That path is still mathematically correct, but it computes the
second-order term through the general Hessian machinery and is often much
slower than a dedicated rule.

Some operations are handled at interpreter level instead of by an ordinary
primitive handler because they form execution boundaries.

- `scan` re-enters Forward Laplacian for the scan body and densifies sparse
  tracked inputs before the loop. Tracing state, interpreter recursion, and
  dense-fallback logging all matter at this boundary.
- `jit` and `pjit` follow the staged-call path: JaQMC first checks for a
  registered handler by string name, then otherwise recurses into the staged
  sub-`jaxpr`. Use
  {func}`~jaqmc.laplacian.primitives.register_function` when the thing you want
  to customize is a named staged subexpression rather than a single primitive.
- `custom_jvp_call` is evaluated through the bound primitive so the user-defined
  JVP rule is preserved instead of being bypassed by direct recursion into the
  inner call body.

For most primitive work, the default mindset should be "choose the smallest
wrapper that matches the primitive" rather than "write a custom handler from
scratch." Those wrappers are the default dense extension path: if tracked inputs
arrive in a sparse form, or if a dense Jacobian first needs materialization
back to the standard `(n, *x.shape)` layout, the wrapper helpers densify before
doing their derivative algebra. Reach for them when you want the smallest
correct rule and sparse retention is not the goal.

Most primitives fit one of the reusable wrapper families:

- {func}`~jaqmc.laplacian.primitives.wrap_linear` for linear operations such as
  reshape, transpose, slicing, and many structure-changing transforms
- {func}`~jaqmc.laplacian.primitives.wrap_elementwise` for unary nonlinear
  elementwise array functions such as `exp`, `log`, `tanh`, or `sin`
- {func}`~jaqmc.laplacian.primitives.wrap_multiplication` for
  multiplication-style bilinear operations
- {func}`~jaqmc.laplacian.primitives.wrap_general` for dense "correct first,
  optimize later" handling
- {func}`~jaqmc.laplacian.primitives.wrap_without_fwd_laplacian` for primitives
  that should drop derivative state and return a plain array

Write a dedicated primitive rule only when those wrappers miss structure that
matters. Typical reasons are:

- the primitive has a known analytic Hessian correction, such as division
- the primitive has special contraction structure, such as `dot_general`
- the primitive can preserve sparse ownership exactly in a way the generic
  dense wrappers cannot

When you do need a dedicated handler,
{func}`~jaqmc.laplacian.primitives.setup_handler` is usually the entry point. A
registry handler does not receive the primitive's ordinary Python signature.
Instead it receives the primitive call as packed `(args, kwargs)`, because
dispatch happens after the interpreter has already collected the equation
inputs. The usual pattern is:

1. call {func}`~jaqmc.laplacian.primitives.setup_handler`
2. return `merged_fwd()` immediately if nothing is tracked
3. compute the tracked result from `lapl_args`

That is what this minimal example is showing:

```python
from jaqmc.laplacian import LapTuple
from jaqmc.laplacian.primitives import setup_handler


def my_handler(args, kwargs):
    merged_fwd, lapl_args = setup_handler(my_primitive.bind, args, kwargs)
    if lapl_args is None:
        return merged_fwd()

    x, = lapl_args.x
    y = merged_fwd(x)
    ...
    return LapTuple(y, grad_y, lapl_y)
```

Conceptually, {func}`~jaqmc.laplacian.primitives.setup_handler` does three jobs:

- separate tracked `LapTuple` inputs from untracked inputs
- build a forward function that replays the primitive on primal values
- verify that dense tracked inputs agree on the same tracked-input basis

If your rule does not preserve sparse payloads explicitly, read the Jacobian
through {attr}`~jaqmc.laplacian.LapTuple.dense_jacobian` or let one of the
dense wrappers handle fallback for you.

## Sparse Retention and Fallback

Sparsity matters because many wavefunction intermediates do not depend on every
tracked coordinate. JaQMC keeps a sparse Jacobian only while that dependence
pattern still fits a simple owner-based model.

Start with the idea of an owner. When you seed sparse tracking with
{func}`~jaqmc.laplacian.make_laplacian_input` and `sparse_axis=...`, you choose
one distinguished input axis. Each entry along that axis is one owner, and the
question becomes: for this output element, which owner or owners does it depend
on?

For example, if `x` has shape `(n_particles, coord_dim)` and you track
`sparse_axis=0`, then each particle row `x[i, :]` is one owner.

```python
def one_particle_feature(x):
    return x[:, 0] ** 2


y = one_particle_feature(x)
# y[i] depends only on x[i, :]
```

This is the one-owner pattern: each output position points back to exactly one
owner.

```python
def pair_feature(x):
    return x[:, None, 0] - x[None, :, 0]


y = pair_feature(x)
# y[i, j] depends only on x[i, :] and x[j, :]
```

This is the two-owner pattern: each output position points back to at most two
owners.

Those two patterns correspond to the sparse families JaQMC can represent
exactly:

- {class}`~jaqmc.laplacian.Local1Jacobian`, where each output position depends
  on one owner entry of the chosen input axis
- {class}`~jaqmc.laplacian.Local2Jacobian`, where each output position depends
  on at most two owner entries of that same axis

As long as a primitive preserves one of those two patterns, the Jacobian can
stay sparse. Once the dependence spreads more broadly, JaQMC needs to densify.
That is the main retention test.

Sparse blocks always carry the full public output shape after a fixed leading
prefix. The layout is `(support_slot, input_coord, *output_shape)`. The
{attr}`~jaqmc.laplacian.Local1Jacobian.output_shape` property (and the matching
`Local2Jacobian` property) is derived directly from `blocks.shape[2:]`. A
shape-changing sparse rule must transform the primal, block payload, and owner
roles in the same output coordinate system while leaving the support/coordinate
prefix untouched. When an operation broadcasts sparse blocks, use
`jax.lax.broadcast_in_dim` (or the shared sparse helpers) so the trailing output
axes match the broadcast primal shape without moving the prefix.

{meth}`~jaqmc.laplacian.Local1Jacobian.with_blocks` (and the matching
{meth}`~jaqmc.laplacian.Local2Jacobian.with_blocks` helper) derives the output
shape from its replacement payload. It revalidates the owner roles against that
new shape, so handlers must transform owner metadata whenever an operation
changes the axis to which owner ids refer.

Dense fallback typically happens for one of three reasons:

- there is no sparse-preserving rule for that primitive, so JaQMC uses a dense
  wrapper
- the primitive has a sparse rule, but the new dependence pattern is no longer
  exactly representable as {class}`~jaqmc.laplacian.Local1Jacobian` or
  {class}`~jaqmc.laplacian.Local2Jacobian`
- the operation needs dense materialization before a generic dense rule can
  continue

`scan` is the main special case worth remembering: sparse tracked inputs are
densified before entering the scan body, so sparse retention is not expected
across that boundary today. If you are debugging an unexpected dense path, start
by asking which of those three fallback reasons you hit before looking for a
bug in the math.

## Troubleshooting

Unexpected dense fallback
: Most often this means the primitive has no sparse-preserving rule, the sparse
  dependence pattern is no longer representable as
  {class}`~jaqmc.laplacian.Local1Jacobian` or
  {class}`~jaqmc.laplacian.Local2Jacobian`, or a generic dense wrapper required
  materialization. Start with the dense wrapper model and the
  primitive-specific rule.

My handler never runs
: First check whether the equation is intercepted by interpreter-level special
  cases such as `scan`, `jit` / `pjit`, or `custom_jvp_call` before ordinary
  primitive dispatch. Start by checking whether the operation is handled at the
  interpreter boundary rather than by the ordinary primitive registry.

Dense basis mismatch
: {func}`~jaqmc.laplacian.primitives.setup_handler` requires all dense tracked
  inputs in one primitive call to share the same leading tracked-input basis.
  If mixed tracked inputs fail before your rule body runs, inspect
  {func}`~jaqmc.laplacian.primitives.setup_handler`.

## What to Read Next

- <project:/api-reference/laplacian.md> for the public transform API and the
  low-level helper entrypoints under {mod}`jaqmc.laplacian.primitives`
- <project:custom-rules.md> if the problem turns out to be helper-local rather
  than primitive-level
- start with {func}`~jaqmc.laplacian.primitives.setup_handler` and the wrapper
  family closest to the primitive you are studying
