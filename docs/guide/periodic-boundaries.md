# Periodic Boundary Conditions

Periodic boundary conditions (PBC) let JaQMC represent an infinite lattice with a
finite simulation cell. This page explains the shared mechanics behind periodic
runs: the supercell picture, the periodic features fed into the wavefunction,
twisted boundary conditions, and, where relevant, Bloch phases.

For a periodic run, keep these three objects separate:

- The **primitive cell** defines the crystal structure and its reciprocal lattice.
- The **supercell** is the finite simulation cell repeated periodically during
  sampling and energy evaluation.
- The **twist** shifts the allowed momenta of that supercell and is the knob used
  for twist averaging.

## Supercell approximation

JaQMC does not simulate an infinite periodic system directly. Instead, it builds a
finite **supercell** by tiling the primitive cell and then repeats that supercell
in every direction. Walkers live in this simulation cell, and any electron that
crosses one face re-enters through the opposite face.

The corresponding Hamiltonian is
$$
\hat{H}_s = -\sum_i \frac{1}{2 m_i} \nabla_i^2 + \frac{1}{2} \sum_i \sum_j \sum_{\mathbf{L}_s}^\prime V_{ij}(\mathbf{r}_i - \mathbf{r}_j + \mathbf{L}_s),
$$
where $\mathbf{L}_s$ runs over supercell lattice translations. The prime means that
the $\mathbf{L}_s = 0$ term is omitted when $i = j$, so a particle does not interact
with itself.

Because this Hamiltonian is invariant under supercell lattice translations,
periodic eigenstates cannot change arbitrarily when a particle crosses the cell
boundary. Instead, translating an electron by a supercell lattice vector can only
change the wavefunction by a phase.

## Periodic input features

In an open system, the raw displacement vector is a reasonable input feature. Under
PBC it is not. Two configurations that differ only by a lattice translation
represent the same physical state, so the feature representation must be
**periodic**. At the same time, the Hamiltonian does not acquire a physical
singularity when a particle crosses one face of the cell and re-enters through the
opposite face, so the wavefunction should remain **smooth** across that boundary.

This smoothness matters because JaQMC differentiates the wavefunction when it
computes gradients and local kinetic-energy terms. A naively wrapped displacement
would be periodic but would still introduce artificial kinks at the boundary.
JaQMC therefore replaces Euclidean distances with smooth periodic features.

### Distance functions (`wf.distance_type`)

JaQMC first projects a relative displacement $\mathbf{r}$ onto reciprocal vectors
$\mathbf{b}_i$,
$$
\omega_i = \mathbf{b}_i \cdot \mathbf{r},
$$
It then applies one of two periodic distance constructions. Each one defines a
scalar periodic distance together with the periodic displacement features used to
represent pair geometry:

- **`tri`** (default) uses trigonometric features directly. Its scalar periodic
  distance is
  $$
  d(\mathbf{r}) =
  \sqrt{
    \sum_{i,j}
    \left[
      \sin(\omega_i)\sin(\omega_j) +
      (1 - \cos(\omega_i))(1 - \cos(\omega_j))
    \right]
    (\mathbf{a}_i \cdot \mathbf{a}_j)
  }.
  $$
  The associated periodic displacement feature concatenates
  $\sum_i \sin(\omega_i)\,\mathbf{a}_i$ and
  $\sum_i \cos(\omega_i)\,\mathbf{a}_i$, producing a 6D vector feature per pair.
  This gives the network a richer periodic representation at the cost of a larger
  feature dimension. The sine and cosine functions are periodic by construction, so
  `tri` does not need an explicit wrap of $\omega_i$.

- **`nu`** uses smooth polynomials after wrapping each $\omega_i$ into
  $[-\pi, \pi]$:
  $$
  f(\omega) = |\omega|\left(1 - \tfrac{1}{4}|\omega/\pi|^3\right), \qquad
  g(\omega) = \omega\left(1 - \tfrac{3}{2}|\omega/\pi| + \tfrac{1}{2}|\omega/\pi|^2\right).
  $$
  The scalar periodic distance is
  $$
  d(\mathbf{r}) =
  \sqrt{
    \sum_i \|\mathbf{a}_i\|^2 f(\omega_i)^2 +
    \sum_{i \ne j} (\mathbf{a}_i \cdot \mathbf{a}_j)\, g(\omega_i)\, g(\omega_j)
  }.
  $$
  The associated periodic displacement feature is
  $\sum_i g(\omega_i)\,\mathbf{a}_i$, so the vector part stays 3D and remains
  close in shape to the open-boundary representation.

Both choices are smooth at the boundary and differentiable everywhere JaQMC needs
them for gradient-based optimization.

### Symmetry expansion (`wf.sym_type`)

The primitive reciprocal basis does not always expose every symmetry-equivalent
direction of the lattice. `wf.sym_type` expands that basis with additional integer
linear combinations before the periodic distance is computed. Both `nu` and `tri`
use this expanded basis.

Available options are:

| Option | Vectors | Use for |
|--------|---------|---------|
| `minimal` | 3 (identity) | No expansion; use only the primitive reciprocal basis |
| `fcc` | 4 | Face-centered cubic lattices; adds the $[1,1,1]$ combination |
| `bcc` | 6 | Body-centered cubic lattices; adds face-diagonal combinations |
| `hexagonal` | 4 | Hexagonal lattices; adds the $[-1,-1,0]$ combination |

Choose the option that matches your lattice symmetry. If your lattice does not fit
one of these presets, `minimal` is the safe default.

(twisted-boundary-conditions)=
## Twisted Boundary Conditions

Ordinary PBC require the many-electron wavefunction to repeat exactly when one
electron is translated by a supercell lattice vector. Twisted boundary conditions
relax that requirement: the wavefunction may pick up a phase instead,
$$
\psi(\dots, \mathbf{r}_i + \mathbf{R}_S, \dots) = e^{i\sum_\alpha \theta_\alpha n_\alpha}\, \psi(\dots, \mathbf{r}_i, \dots),
$$
where $\mathbf{R}_S = \sum_\alpha n_\alpha \mathbf{L}_{S\alpha}$ is a supercell
translation written in the supercell basis vectors $\mathbf{L}_{S\alpha}$.

A useful way to read this is in momentum space. For a one-dimensional box of length
$L$, the allowed momenta become
$$
k_n = \frac{2\pi n + \theta}{L},
$$
instead of the ordinary periodic grid $k_n = 2\pi n / L$. A nonzero twist shifts
the entire momentum mesh.

JaQMC exposes this shift through `system.twist`, a three-component vector in
fractional coordinates of the **supercell reciprocal basis**. The default
`[0, 0, 0]` gives ordinary periodic boundary conditions.

In QMC, a common practice is **twist averaging**: run the calculation at many twist
values and average the observables,
$$
E \approx \frac{1}{N_\theta}\sum_{\theta} E(\theta).
$$
This greatly reduces one-body finite-size errors such as shell effects in the
kinetic energy. It does not remove every finite-size error, so long-range Coulomb
and exchange-correlation effects still need separate correction or extrapolation.

JaQMC does not automate twist averaging. To use it, run separate training and
evaluation jobs at different `system.twist` values, then average the evaluation
results afterward.

(bloch-phases-in-the-wavefunction)=
## Bloch Phases in the Wavefunction

Some periodic wavefunctions, including JaQMC's current
<project:../systems/solid/index.md> workflow, use Bloch phases so that the
wavefunction transforms by a phase under lattice translations instead of
remaining strictly unchanged.

In JaQMC's current implementation, the twist enters the wavefunction through the orbital
k-points. Compared with an open-boundary wavefunction, it differs in three ways:

1. **Complex orbitals.** Each orbital has independent real and imaginary parts,
   allowing the model to represent complex-valued Bloch states.

2. **A folded k-point mesh.** The current implementation uses the primitive-cell
   k-points that fold to $\Gamma$ in the simulation supercell. That folded mesh
   has size $|\det(S)|$, which JaQMC exposes as `system.scale` (see
   [Supercell Expansion](../systems/solid/index.md#supercell-expansion)). The final
   orbital list also accounts for spin channels and any additional occupied bands
   at a given k-point.

3. **Bloch phase multiplication.** Each orbital is multiplied by a plane-wave phase
   factor
   $$
   \tilde{\phi}_j(\mathbf{r}_i) = \phi_j(\mathbf{r}_i)\, e^{i\mathbf{k}_j \cdot \mathbf{r}_i},
   $$
   where $\mathbf{k}_j$ is the k-point assigned to orbital $j$. If `system.twist`
   is nonzero, JaQMC shifts the folded mesh before assigning those orbital
   k-points.

Because the orbital matrix is complex, sampled energy estimates can carry a small
imaginary component. Its expectation value vanishes, so the physically meaningful
energy is the real part.
