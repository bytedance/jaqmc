(ext-runtime-data-conventions)=
# Runtime Data Conventions

```{note}
New to JaQMC extension work? Read [the default convention](#ext-runtime-data-default)
and [the running example](#ext-runtime-data-example) first. If you are only writing a wavefunction or a
per-walker estimator, you can usually stop there and skip
[the advanced batching section](#ext-runtime-data-advanced).
```

This page is the reference for JaQMC's built-in runtime data convention. In
most cases, you only need the core mental model:

- Most wavefunctions and per-walker estimator logic see one walker's
  {class}`~jaqmc.data.Data` object at a time.
- Built-in apps usually store walker-dependent particle coordinates in a field
  such as `electrons`, and for one walker that field usually has shape
  `(n_particles, ndim)`.
- {class}`~jaqmc.data.BatchedData` mostly matters when you are writing
  samplers, workflows, or other low-level code that manipulates whole batches
  of walkers directly.

If you came here from <project:writing-workflows.md> or <project:wavefunctions.md>, this
page explains the runtime data contract those guides build on.

If you are writing a sampler, workflow, or other batch-aware plumbing, keep
reading into the advanced section. Otherwise, the next two sections are the
main thing to learn.

## What `Data` Is For

{class}`~jaqmc.data.Data` is JaQMC's container for structured runtime inputs.
For a beginner, the simplest way to think about it is: "`Data` holds the arrays
my components read while the run is executing."

Most high-level extension hooks use `Data` in its single-walker form. A
wavefunction or `evaluate_local` implementation usually receives one walker's
values in that object.

It is not limited to electron coordinates. Built-in apps already use
{class}`~jaqmc.data.Data` for other system inputs such as atomic positions and
charges. That is why a molecular {class}`~jaqmc.data.Data` object contains
`electrons`, `atoms`, and `charges`, not just one electron array.

Lower-level framework code can also take the same `Data` shape and pair it with
batching metadata using {class}`~jaqmc.data.BatchedData`, but you usually do
not need to think about that at first.

```{tip}
Put values that change at runtime in {class}`~jaqmc.data.Data`, such as
particle coordinates.
Values that determine array shapes or control flow, such as `n_particles` or
`ndim`, should usually stay in config so JAX can treat them as static.
```

(ext-runtime-data-default)=
## Default Convention

With that mental model in place, start from the single-walker view.

- Most wavefunction and per-walker estimator hooks treat a
  {class}`~jaqmc.data.Data` object as one walker's structured runtime inputs.
- Built-in apps usually store walker-dependent particle coordinates in a field
  such as `electrons`.
- The built-in recommendation is that such a field has shape
  `(n_particles, ndim)` for one walker.

If you only remember one rule, remember this one: inside a wavefunction,
`data.electrons` usually means one walker's particle coordinates, not a batch.

This convention also applies to other walker-dependent particle fields, not
just `electrons`. The built-in apps and most examples use `electrons`, but the
same pattern can be used for other particle fields if your custom components
agree on it.

(ext-runtime-data-example)=
## Running Example

For a molecule-style `Data` object, built-in code typically uses:

- `data.electrons`: `(n_particles, 3)`
- `data.atoms`: `(n_atoms, 3)`
- `data.charges`: `(n_atoms,)`

Wavefunctions usually consume exactly this single-walker view. That is why
{meth}`~jaqmc.wavefunction.Wavefunction.__call__` implementations can treat
`data.electrons` as one walker's particle coordinates rather than a batch.

The same idea appears in the built-in Hall app even though the coordinate system
changes: each walker still carries one `electrons` field, but its per-particle
coordinates are spherical `(theta, phi)` pairs instead of Cartesian triples.

(ext-runtime-data-advanced)=
## Advanced: `BatchedData` and `fields_with_batch`

This section is for custom samplers, estimators, and workflow authors who need
to reason about walker batches explicitly.

If you are only writing a wavefunction or per-walker estimator logic, you can
usually stop at the single-walker convention above. The rest of this page is
about the lower-level batch representation used by framework plumbing.

Start with the simplest picture: `BatchedData` means "one runtime data object,
plus information about which fields are batched over walkers."

It is not a Python list of per-walker `Data` objects. It is one
{class}`~jaqmc.data.Data`-shaped pytree plus metadata describing which fields
currently carry a leading walker axis.

{class}`~jaqmc.data.BatchedData` stores two things:

- `data`: a {class}`~jaqmc.data.Data` object
- `fields_with_batch`: metadata naming which fields carry a leading walker axis

For the built-in `electrons` convention, the single-walker and batched views
usually line up like this:

| Object | Typical shape | Meaning |
|------|------|------|
| `data.electrons` | `(n_particles, ndim)` | One walker's particle coordinates |
| `batched_data.data.electrons` | `(batch, n_particles, ndim)` | One batched particle field |
| `batched_data.data.atoms` | `(n_atoms, ndim)` | Shared across walkers |
| `batched_data.data.charges` | `(n_atoms,)` | Shared across walkers |
| `batched_data.fields_with_batch` | `["electrons"]` | Declares which fields carry the walker axis |

A field is "batched" when both of these are true:

- it has a leading walker axis in its actual array shape
- it is listed in `fields_with_batch`

An "unbatched" field is simply shared across walkers in this `BatchedData`
object. It does not need to be scalar. In built-in apps, fields like `atoms`,
`charges`, and `primitive_atoms` are often full arrays that stay shared across
walkers.

{attr}`~jaqmc.data.BatchedData.fields_with_batch` is not just documentation.
JaQMC uses it to drive real framework behavior:

- {attr}`~jaqmc.data.BatchedData.vmap_axis` tells `jax.vmap` which fields map
  over walkers
- {attr}`~jaqmc.data.BatchedData.partition_spec` marks which fields are sharded
  over the batch axis
- {meth}`~jaqmc.data.BatchedData.check` validates that batched fields agree on
  batch size
- {meth}`~jaqmc.data.BatchedData.fully_batched_data` can broadcast shared
  fields across the batch
- {meth}`~jaqmc.data.BatchedData.all_gather` gathers batched fields from
  sharded execution

Built-in molecule, solid, and Hall apps all currently batch only `electrons`.
That is the convention to copy unless you have a concrete reason to do
something else.

You do not always have to construct {class}`~jaqmc.data.BatchedData` yourself.
Simple examples can return plain {class}`~jaqmc.data.Data` from `data_init`,
and JaQMC will wrap it using the usual built-in assumption that `electrons` is
the batched particle field. In that case the returned `Data` object is already
batch-shaped on `electrons`; JaQMC is only adding the `fields_with_batch`
metadata. Return explicit
{class}`~jaqmc.data.BatchedData` when your custom layout needs different
batching metadata.

## Other Particle Fields and Custom Layouts

The built-in convention is a strong recommendation, not a hard framework law.

If you want to use a different particle field name or a different layout, you
can. The requirement is consistency: your wavefunction, sampler, estimators, and
any workflow code that reads or writes those fields must all agree on the same
contract.

Use the built-in convention when you want to match JaQMC examples and defaults.
Deviate from it only when the alternative is clearly better for your custom
system or method.
