# Pseudo-Hamiltonian (PH) pseudopotentials

For configuration options, see the estimator sections of each system's config reference ([molecule](#molecule-estimators)).

Pseudo-Hamiltonian (PH) pseudopotentials are an alternative pseudopotential
family that sits parallel to semi-local [ECP](project:ecp.md)
pseudopotentials. Here, `ECP` refers specifically to the semi-local effective
core potential family, while `PH` refers to the local Pseudo-Hamiltonian
family. Like an ECP, PH replaces the explicit treatment of core electrons.
Unlike an ECP, PH does *not* introduce a nonlocal angular-momentum projector
($|lm\rangle\langle lm|$ in the ECP form); instead, the angular content of the
pseudopotential is absorbed into a position-dependent kinetic operator. As a
result, PH energy evaluation needs no spherical quadrature, no random
rotation, and no localization approximation, and PH is significantly cheaper
to evaluate than the corresponding ECP while delivering the same level of
many-body accuracy on the supported elements.

The implementation here is based on [Bennett et al., *J. Chem. Theory Comput.* **18**, 828–839 (2022)](https://doi.org/10.1021/acs.jctc.1c00992) and the locality-error-free follow-up [Ichibha et al., *J. Chem. Phys.* **159**, 164114 (2023)](https://doi.org/10.1063/5.0175381). The integration with neural-network QMC, including the forward-Laplacian evaluation strategy used by default, follows [Fu et al., arXiv:2505.19909 (2025)](https://arxiv.org/abs/2505.19909).

## Supported elements

PH is currently available for the 3d transition metals **Cr, Mn, Fe, Co, Ni, Cu, Zn** (from Bennett et al. 2022 and Ichibha et al. 2023) and for **S** (from Fu et al. 2025). The radial reference tables are bundled with JaQMC, so no external download is required. Selecting PH for any other element raises a configuration error at startup.

## Configuration

PH is selected per element in the same `system.pp` mapping used for ECPs:

```yaml
system:
  pp:
    Fe: ph
    Li: ccecp
```

In the example above, every `Fe` atom uses PH, every `Li` atom uses the `ccecp` ECP, and any other element is treated all-electron. PH atoms and ECP atoms may coexist in the same system.

## Kinetic backend

On PH runs the kinetic energy is computed by the PH estimator rather than by a standalone {class}`~jaqmc.estimator.kinetic.EuclideanKinetic`. The evaluation strategy is selected via `estimators.energy.ph.kinetic_backend`:

- `forward_laplacian` (default) — Forward-mode Laplacian evaluation via [folx](https://github.com/microsoft/folx), following [Fu et al. (2025)](https://arxiv.org/abs/2505.19909). This is the fast path and is recommended for production runs.
- `standard` — Reverse-mode evaluation via `jax.grad` / `jax.hessian`, following [Bennett et al. (2022)](https://doi.org/10.1021/acs.jctc.1c00992). Slower than `forward_laplacian` but useful as a cross-check and in environments where the forward-Laplacian path is unavailable.

:::{note}
On PH runs the kinetic backend is controlled exclusively by `estimators.energy.ph.kinetic_backend`. Setting any `estimators.energy.kinetic.*` key on a PH system is an error and will fail the unused-config check at startup. A startup log line reports the active PH backend.
:::

For the algorithm behind each backend, see [Backends](#ph-backends) below.

## Outputs

On a PH run the energy stats include:

- `energy:kinetic` — the PH-modified kinetic contribution. This replaces the ordinary kinetic energy used on non-PH runs.
- `energy:ph` — the local PH potential contribution, summed over PH atoms.
- `energy:potential` — the bare electron-nucleus and electron-electron Coulomb terms, unchanged from non-PH runs.

The total energy is the sum of these, plus any semi-local ECP contributions
when PH and ECP coexist.

## Limitations

- PH is currently wired only for the molecule workflow. Solid systems are not yet supported.

(ph-formulas)=
## Local energy formulas

The PH local energy contribution per walker splits into two pieces: a *derivative* term that replaces the ordinary kinetic energy (reported as `energy:kinetic`) and a short-range *zero-order* residual (reported as `energy:ph`). Both are summed over electrons and over PH-treated atoms.

Throughout this section, $\psi$ denotes the trial wavefunction (the same one the rest of the estimators see). For one electron $i$ at position $\mathbf{x}_i$, the PH derivative term is

$$
E_{\text{PH}, i}
= -\mathrm{Tr}(M_i H_i)
  - g_i^\top M_i g_i
  + b_i^\top g_i
$$

where $g_i = \nabla_{\mathbf{x}_i} \log\psi$ is the per-electron gradient of $\log\psi$, $H_i$ is the diagonal $3\times 3$ Hessian block of $\log\psi$ with respect to $\mathbf{x}_i$, $M_i$ is the per-electron mass matrix, and $b_i$ is the per-electron first-order vector ([Bennett et al. 2022](https://doi.org/10.1021/acs.jctc.1c00992)). Together the three terms are equivalent to $-\tfrac{1}{2} \nabla \cdot (d \nabla \psi) / \psi$ in the original notation, with $d = 2M$: $-\mathrm{Tr}(M_i H_i) - g_i^\top M_i g_i$ is the symmetric kinetic-like piece, and $b_i^\top g_i$ is the drift correction from $-\nabla \cdot M$. The two backends documented in [Backends](#ph-backends) compute this same quantity by different routes.

(ph-mass-matrix)=
### Mass matrix $M$

Starting from Bennett's PH diffusion tensor $d(r) = I + 2 \left(r^2 I - \mathbf{r} \mathbf{r}^\top\right) v_{L^2}(r)$ and using $M = d / 2$, the per-electron mass matrix is

$$
M_i = \tfrac{1}{2} I + \sum_{a \in \text{PH}} \ell_2(r_{ia}) \left(r_{ia}\, I - \frac{\mathbf{r}_{ia} \mathbf{r}_{ia}^\top}{r_{ia}}\right)
$$

where $\mathbf{r}_{ia} = \mathbf{x}_i - \mathbf{R}_a$, $r_{ia} = |\mathbf{r}_{ia}|$, and $\ell_2(r) = r \cdot v_{L^2}(r)$ (the bundled XML tables store the radial pseudopotential in this $r \cdot V$ form). The sum runs only over PH-treated atoms.

$M_i$ must remain positive definite for the PH derivative to be well defined. If interpolated tables drive an eigenvalue non-positive, both backends propagate NaN into `energy:kinetic` rather than silently flooring the eigenvalue; see [Backends](#ph-backends) for the per-backend mechanism.

(ph-first-order-vector)=
### First-order vector $b$

The first-order vector is the divergence of the L²-only part of the mass tensor,

$$
b_i = -\nabla \cdot \left(M_i - \tfrac{1}{2} I\right)
    = \sum_{a \in \text{PH}} 2\, v_{L^2}(r_{ia})\, \mathbf{r}_{ia}
    = \sum_{a \in \text{PH}} 2\, \frac{\ell_2(r_{ia})}{r_{ia}}\, \mathbf{r}_{ia}.
$$

The closed form on the right is what the production backend uses directly; the standard backend instead differentiates $M - I/2$ numerically with `jax.jacfwd` (see [Backends](#ph-backends)).

### Zero-order residual

The zero-order PH contribution is the short-range residual local channel,

$$
E_{\text{PH}, 0}
= \sum_{i} \sum_{a \in \text{PH}}
  \left[ \tilde v_{\text{loc}}(r_{ia}) + \frac{Z_a}{r_{ia}} \right],
$$

where $\tilde v_{\text{loc}}$ is the bundled paper-form local channel and $Z_a$ is the PH effective valence charge for atom $a$. The XML loader stores the table as $r \cdot \tilde v_{\text{loc}}(r) + Z_a$, so the estimator can recover $E_{\text{PH}, 0}$ as $\sum \text{loc\_data}(r_{ia}) / r_{ia}$ without re-introducing $Z_a$ at evaluation time.

The $+ Z_a / r_{ia}$ piece cancels the bare electron-nucleus Coulomb $-Z_a / r_{ia}$ that `potential_energy` already supplies for every atom, so

$$
\sum_{i, a \in \text{PH}} \tilde v_{\text{loc}}(r_{ia})
  = \verb|energy:ph| + \verb|energy:potential|\big|_{\text{PH atoms}}
$$

recovers the paper-form local channel without any masking between the two estimators. This mirrors how {class}`~jaqmc.estimator.ecp.estimator.ECPEnergy` composes additively with `potential_energy`.

(ph-backends)=
## Backends

Both backends compute the same operator decomposition shown in [Local energy formulas](#ph-formulas); they differ only in *how* the derivatives of $\log\psi$ are obtained. The user-facing knob is documented in [Kinetic backend](#kinetic-backend) above.

(ph-backend-fl)=
### Forward-Laplacian backend

The production path, following the NNQMC + PH + Forward-Laplacian construction of [Fu et al. (2025)](https://arxiv.org/abs/2505.19909). It assembles the PH derivative term in a single forward-Laplacian pass by folding the per-electron Cholesky factor of $M$ into the shifted-coordinate map.

Cholesky-decompose $M_i = L_i L_i^\top$ per electron and define

$$
g(\mathbf{y}) = \log\psi\!\left(\dots,\, \mathbf{x}_i + L_i \mathbf{y}_i,\, \dots\right)
\quad \text{evaluated at } \mathbf{y} = 0.
$$

Then

$$
\nabla_{\mathbf{y}} g\big|_{\mathbf{y}=0} = L_i^\top \nabla_{\mathbf{x}_i} \log\psi,
\qquad
\mathrm{Tr}\!\left(\nabla_{\mathbf{y}}^2 g\right)\big|_{\mathbf{y}=0}
  = \mathrm{Tr}(L_i^\top H_i L_i) = \mathrm{Tr}(M_i H_i),
$$

so a single [folx](https://github.com/microsoft/folx) `forward_laplacian` evaluation of $g$ yields both $-\mathrm{Tr}(M_i H_i)$ and $-g_i^\top M_i g_i$ summed over electrons. The remaining $b_i^\top g_i$ term is added using the closed-form $b_i$ from [First-order vector b](#ph-first-order-vector) and a separate `jax.grad` call for $\nabla_{\mathbf{x}} \log\psi$.

If $M_i$ is not positive definite at the evaluation point, `jnp.linalg.cholesky` returns NaN and `energy:kinetic` becomes NaN — the expected loud failure mode for this backend.

(ph-backend-standard)=
### Standard backend

The paper-faithful reverse-mode reference path, following [Bennett et al. (2022)](https://doi.org/10.1021/acs.jctc.1c00992). It assembles each term in $E_{\text{PH}, i} = -\mathrm{Tr}(M_i H_i) - g_i^\top M_i g_i + b_i^\top g_i$ directly:

- $g_i$ via `jax.grad` of $\log\psi$.
- $H_i$ via `jax.hessian` of $\log\psi$ (the per-electron diagonal $3\times 3$ block; cross-electron blocks do not enter the PH derivative term).
- $M_i$ assembled in the closed form of [Mass matrix M](#ph-mass-matrix).
- $b_i$ obtained as $-\nabla \cdot (M - I/2)$ via `jax.jacfwd`, which is algebraically the same as the analytic identity used by the forward-Laplacian backend but reaches it through a different code path.
- The three contractions are then performed by explicit `jnp.einsum` calls.

Non-PD assembly is caught with an explicit eigenvalue guard on $M_i$ that returns a NaN-filled matrix, matching the loud-failure behavior of the forward-Laplacian backend by a different mechanism.

This backend is slower than the forward-Laplacian path but easier to read and serves as the math anchor exercised by the parity test; performance work and pathological-input hardening live in the forward-Laplacian backend.

## See also

- [Pseudopotentials (ECP)](project:ecp.md) — the sibling semi-local
  pseudopotential family.
- [Kinetic energy](project:kinetic.md) — the standalone kinetic estimator used on non-PH systems.
- Configuration: [Molecule](#molecule-estimators)
- API: {class}`~jaqmc.estimator.ph.PHEnergy`
