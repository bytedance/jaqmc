# Periodic Boundary Conditions

JaQMC supports periodic boundary conditions (PBC) for systems defined on a lattice. This page explains the cross-cutting mechanics that apply to any PBC-enabled workflow: how distances are computed, how Bloch phases work, and how twisted boundary conditions reduce finite-size errors.

Currently, the <project:../systems/solid/index.md> workflow uses PBC.

## Distance Functions (`distance_type`)

In periodic systems, the distance between two points is not unique — each atom has infinitely many periodic images. JaQMC uses smooth periodic distance functions that respect the lattice symmetry without requiring an explicit sum over images.

Two distance functions are available, selected via `wf.distance_type`:

**`nu`** (default) — A polynomial approximation to the periodic distance. For each expanded reciprocal lattice vector $\mathbf{b}_i$, it computes $\omega_i = \mathbf{r} \cdot \mathbf{b}_i$ (wrapped to $[-\pi, \pi]$) and applies smooth periodic functions:

$$f(\omega) = |\omega|\left(1 - \tfrac{1}{4}|\omega/\pi|^3\right), \qquad g(\omega) = \omega\left(1 - \tfrac{3}{2}|\omega/\pi| + \tfrac{1}{2}|\omega/\pi|^2\right)$$

These feed into a distance formula that combines contributions from all lattice directions. The relative displacement vector is reconstructed as $\sum_i g(\omega_i)\,\mathbf{a}_i$, giving 3D features per pair.

**`tri`** — A trigonometric distance using $\sin(\omega_i)$ and $\cos(\omega_i)$ directly. This produces 6D features per pair (sine and cosine components), giving the network richer symmetry information at the cost of a larger feature dimension.

Both functions are smooth and periodic, so the neural network can differentiate through them for gradient computation. The default `nu` works well for most systems.

## Symmetry Expansion (`sym_type`)

The primitive reciprocal lattice basis may not expose all symmetry-equivalent directions of the crystal. The `sym_type` option expands the reciprocal basis with additional integer linear combinations, giving the distance functions access to more periodic "views" of the geometry.

Available options for `wf.sym_type`:

| Option | Vectors | Use for |
|--------|---------|---------|
| `minimal` | 3 (identity) | No expansion — only the primitive reciprocal basis |
| `fcc` | 4 | Face-centered cubic lattices (adds the $[1,1,1]$ combination) |
| `bcc` | 6 | Body-centered cubic lattices (adds face-diagonal combinations) |
| `hexagonal` | 4 | Hexagonal lattices (adds the $[-1,-1,0]$ combination) |

The expanded basis is used by both `nu` and `tri` distance functions. Choose the option that matches your crystal symmetry. For lattices that don't fit any preset, `minimal` is a safe default.

(twisted-boundary-conditions)=
## Twisted Boundary Conditions

The `twist` parameter is a fractional k-point in the primitive cell's reciprocal space — a vector in $[0, 1)^3$ that shifts the Bloch phases of all orbitals. Physically, it controls the boundary condition that electrons satisfy when wrapping around the simulation cell:

$$\psi(\mathbf{r} + \mathbf{R}) = e^{i\mathbf{k} \cdot \mathbf{R}}\,\psi(\mathbf{r})$$

where $\mathbf{R}$ is a lattice vector and $\mathbf{k}$ is derived from the twist. At `twist = [0, 0, 0]` (the $\Gamma$ point), the wavefunction is strictly periodic. Nonzero twist values impose a phase shift at the cell boundary.

Twist averaging — running multiple simulations at different twist values and averaging the energies — reduces finite-size errors caused by the discrete k-point sampling of the Brillouin zone. JaQMC does not automate twist averaging, but you can run separate training jobs at different twist values and average the evaluation energies.

(bloch-phases-in-the-wavefunction)=
## Bloch Phases in the Wavefunction

PBC wavefunctions differ from open-boundary wavefunctions in three key ways:

1. **Complex orbitals.** Each orbital has both real and imaginary components ($\phi_\text{real} + i\,\phi_\text{imag}$), enabling the wavefunction to represent complex-valued Bloch states.

2. **Multiple k-points.** In a supercell, each orbital is associated with a k-point from the folded Brillouin zone. The number of k-points equals the supercell scale factor.

3. **Bloch phase multiplication.** Each orbital is multiplied by a plane-wave phase factor:

$$\tilde{\phi}_j(\mathbf{r}_i) = \phi_j(\mathbf{r}_i) \cdot e^{i\mathbf{k}_j \cdot \mathbf{r}_i}$$

where $\mathbf{k}_j$ is the k-point assigned to orbital $j$. This ensures the wavefunction transforms correctly under lattice translations, as required by Bloch's theorem. The k-points include any shift from the `twist` parameter.

Because the wavefunction is complex-valued, the reported `total_energy` has a small imaginary component. This is a finite-sampling artifact whose expectation value vanishes — only the real part is physically meaningful.
