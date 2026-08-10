# Moire

## Background

Moire materials formed by stacking two-dimensional layers with a small twist
angle host a rich landscape of strongly correlated topological phases. In
twisted bilayer MoTe₂ (tMoTe₂), strong spin-valley locking separates the top
valence band from the rest, and the low-energy physics is captured by a
two-component **continuum model** in which the two components label the top and
bottom layers (a **layer pseudospin**). The interplay of the intra-layer moire
potential and the inter-layer tunneling forms an effective skyrmion field that
gives the band a non-trivial topology, so tMoTe₂ realizes integer Chern
insulators, Z₂ topological insulators, and — at fractional fillings — fractional
Chern insulators (FCI) competing with charge density waves (CDW).

The `jaqmc moire` app implements the deep-learning variational Monte Carlo
approach of [Li et al., "Deep Learning Sheds Light on Integer and Fractional
Topological Insulators" (arXiv:2503.11756)](https://arxiv.org/abs/2503.11756).
A neural-network many-body wavefunction is optimized with VMC over both the
electron positions and a continuous layer-pseudospin variable, which lets a
single ansatz capture band mixing without any Landau-level or band projection.

For each physical spin channel the effective single-particle Hamiltonian is a
$2\times 2$ operator in the layer basis. For the spin-up valence electrons
(after a particle-hole transformation),

$$
H_\uparrow(\mathbf{r}) =
\begin{pmatrix}
\dfrac{(-i\nabla - \mathbf{K}_+)^2}{2m} + \Delta_b(\mathbf{r}) & \Delta_T(\mathbf{r}) \\[2mm]
\Delta_T^*(\mathbf{r}) & \dfrac{(-i\nabla - \mathbf{K}_-)^2}{2m} + \Delta_t(\mathbf{r})
\end{pmatrix},
$$

and the spin-down Hamiltonian is fixed by time-reversal symmetry: the valley
momenta flip sign and the off-diagonal tunneling is conjugated,

$$
H_\downarrow(\mathbf{r}) =
\begin{pmatrix}
\dfrac{(-i\nabla + \mathbf{K}_+)^2}{2m} + \Delta_b(\mathbf{r}) & \Delta_T^*(\mathbf{r}) \\[2mm]
\Delta_T(\mathbf{r}) & \dfrac{(-i\nabla + \mathbf{K}_-)^2}{2m} + \Delta_t(\mathbf{r})
\end{pmatrix}.
$$

The intra-layer potentials $\Delta_{b/t}$ (diagonal) and the inter-layer
tunneling $\Delta_T$ (off-diagonal) are first-harmonic moire fields,

$$
\Delta_{b/t}(\mathbf{r}) = -2V \sum_{i=1,3,5} \cos(\mathbf{g}_i \cdot \mathbf{r} \pm \delta),
\qquad
\Delta_T(\mathbf{r}) = \omega\left(1 + e^{i\mathbf{g}_2 \cdot \mathbf{r}} + e^{i\mathbf{g}_3 \cdot \mathbf{r}}\right),
$$

where the $+\delta$ sign applies to the bottom layer $\Delta_b$ and $-\delta$ to
the top layer $\Delta_t$. The reciprocal vectors and valley momenta are

$$
\mathbf{g}_i = \frac{4\pi}{\sqrt{3}\,a_M}\left(\cos\frac{\pi(i-1)}{3}, \sin\frac{\pi(i-1)}{3}\right),
\qquad
\mathbf{K}_+ = \frac{\mathbf{g}_1 + \mathbf{g}_2}{3},
\qquad
\mathbf{K}_- = \frac{\mathbf{g}_1 + \mathbf{g}_6}{3},
$$

with moire length $a_M = a_0 / [2\sin(\theta/2)]$ at twist angle $\theta$. The
full many-body Hamiltonian adds the electron-electron Coulomb interaction in a
uniform neutralizing background,

$$
H_{\text{total}} = \sum_i \left[ H_\uparrow(\mathbf{r}_i) + H_\downarrow(\mathbf{r}_i) \right]
  + \frac{1}{2}\sum_{i \neq j} v_E(\mathbf{r}_i - \mathbf{r}_j)
  + \frac{N_e}{2} v_M,
$$

where $v_E$ is the 2D Ewald-summed Coulomb interaction with relative dielectric
constant $\epsilon$, and $v_M$ is the Madelung constant (image self-interaction).
The default model parameters for tMoTe₂ are $a_0 = 0.352$ nm, $m = 0.62\,m_e$,
$V = 11.2$ meV, $\omega = -13.3$ meV, $\delta = -91^\circ$, and $\epsilon = 10$.

When the local energy is evaluated, $H_{\text{total}}$ separates into four
physical contributions, and the app names them accordingly:

- the **kinetic** energy, the Euclidean Laplacian
  $-\tfrac{1}{2m}\sum_i \nabla_i^2$ shared by both layers;
- the **valley-momentum shift** (also called SOC), the valley-dependent
  correction that promotes the plain Laplacian to the shifted operators
  $(-i\nabla - \eta\,\mathbf{K}_\pm)^2/2$ — with the physical-spin sign
  $\eta=+1$ for spin-up and $\eta=-1$ for spin-down — and couples to the layer
  pseudospin;
- the **moire potential**, the single-particle $2\times2$ field $\Delta$ that
  groups the intra-layer potentials $\Delta_{b/t}$ on the diagonal with the
  inter-layer tunneling $\Delta_T$ off the diagonal;
- the **Coulomb** interaction between electrons, $\tfrac12\sum_{i\neq j}
  v_E(\mathbf{r}_i-\mathbf{r}_j) + \tfrac{N_e}{2}v_M$, where the neutralizing
  background and Madelung self-interaction $\tfrac{N_e}{2}v_M$ are folded into
  the 2D Ewald sum rather than added separately.

User-facing energies are in **meV**. Internally the estimators work in
dimensionless simulation units and rescale by prefactors derived from
`moire_lattice_constant_nm`, `effective_mass`, and `dielectric_constant`
(kinetic $\propto \text{Ha}/L^2$, Coulomb $\propto \text{Ha}/(\epsilon L)$).

```{note}
The model is fixed by only the first-harmonic terms above, parameterized by $V$,
$\omega$, $\delta$, $m$, and $\epsilon$ (exposed as `v1_mev`, `omega1_mev`,
`phi1_deg`, `effective_mass`, and `dielectric_constant`). See the
[configuration reference](train.md) for the full list of config keys.
```

## Basic Usage

The default system is a fractional two-thirds-filling config ($n=-2/3$): an
8-electron spin-polarized $3\times4$ supercell (`electron_spins=[8,0]`) built
from a triangular primitive cell. Select a different system by overriding
`system.*` fields. Some representative systems:

- integer filling $n=-2$: an 18-electron $3\times3$ supercell built from a
  rectangular primitive cell
  (`system.electron_spins="[9,9]" system.nelec=18 system.supercell_matrix="[[3,0],[0,3]]"`)
- fractional filling $n=-1/3$: a 9-electron spin-polarized non-diagonal
  $(3\sqrt{3})\times(3\sqrt{3})$ supercell, $\det S = 27$
  (`system.electron_spins="[9,0]" system.nelec=9 system.supercell_matrix="[[6,3],[3,6]]"`)

Train:

```bash
jaqmc moire train workflow.save_path=./runs/moire-train
```

Evaluate:

```bash
jaqmc moire evaluate workflow.save_path=./runs/moire-eval \
  workflow.source_path=./runs/moire-train
```

Override any `system.*` key inline (e.g. `system.supercell_matrix="[[2,1],[1,2]]"`)
or pass `--yml your.yml` to load a full config file. For shared workflow
mechanics such as `--dry-run`, resume, and output paths, see
<project:../../guide/running-workflows.md>.

## How It Works

Each walker carries structured 2D state: spatial coordinates in `positions` with
shape `(n_elec, 2)` and continuous pseudo-spin angles in `spin_coords` with shape
`(n_elec,)`. Training and evaluation use a joint Metropolis-Hastings proposal so
that spatial and spin moves are proposed and accepted together.

### Continuous spin

Because the two-component Hamiltonian does not commute with the $z$-component of
the layer pseudospin, the layer index must be treated as a dynamical variable.
Sampling a discrete layer index directly causes high rejection rates, so the app
uses the **continuous spin** technique that maps the discrete layer to a
continuous angle $s \in [0, 2\pi)$. Throughout the app this sampled "spin" is
always the **layer pseudospin**, distinct from the fixed physical electron spin
that labels the $H_\uparrow$ / $H_\downarrow$ channels. Each single-particle
orbital becomes

$$
\phi(\mathbf{r}, s) = \phi_b(\mathbf{r})\,e^{is} + \phi_t(\mathbf{r})\,e^{-is},
$$

where $\phi_{b/t}$ are the bottom/top spatial orbitals. In the neural network
these are promoted to correlated orbitals
$\phi_{b/t}(\mathbf{r}_i; \mathbf{r}_{\neq i})$. For $L$ layers,
`MoireWavefunction.layer_components` exposes `(phi, chi, phi0)`: layer orbitals
$\phi_{\ell,dij}$, sampled phase amplitudes $\chi_{i\ell}$, and the identity
orbital matrix

$$
\Phi^0_{dij} = \sum_{\ell=1}^L \chi_{i\ell}\phi_{\ell,dij}.
$$

`phi0` is exactly the matrix returned by `MoireWavefunction.orbitals()` and is
used to evaluate $\Psi=\sum_d\det\Phi^0_d$. This layer representation is
generic in $L$; the current `MoirePotential` and `MoireSOC` estimators construct
concrete bilayer $2\times2$ matrices locally. `MoirePotential` contracts the
physical matrix $\Delta(\mathbf{r}_i)$ above with the layer components and
evaluates its local determinant ratio against $\Phi^0$.

(moire-soc)=
### Valley-momentum shift (SOC)

Each layer lives at a different valley of the underlying monolayers, so its
electrons carry a built-in momentum — $\mathbf{K}_+$ on the bottom layer and
$\mathbf{K}_-$ on the top, where the valley index $\pm$ is the layer pseudospin.
Time reversal maps the spin-up valleys onto the spin-down ones, so the two
physical spins see opposite shifts: with the physical-spin sign $\eta$
introduced above, the built-in momentum is $\eta\,\mathbf{K}_\pm$. This is what
promotes the ordinary kinetic energy $-\nabla^2/2$ to the shifted operators
$(-i\nabla-\eta\,\mathbf{K}_\pm)^2/2$ on the diagonal of $H_\uparrow$ and
$H_\downarrow$. Expanding the square,

$$
\frac{(-i\nabla - \eta\,\mathbf{K}_\pm)^2}{2}
= -\frac{\nabla^2}{2}
  + i\,\eta\,\mathbf{K}_\pm\cdot\nabla
  + \frac{|\mathbf{K}_\pm|^2}{2},
$$

separates the layer-independent Laplacian — already handled by the kinetic
estimator — from a valley-dependent remainder that `MoireSOC` evaluates.

The interesting piece is the linear term $i\,\eta\,\mathbf{K}_\pm\cdot\nabla$.
Because the two layers sit at opposite valleys, this momentum is **tied to the
layer pseudospin**: it points one way on the bottom layer and the other way on
the top. Splitting each layer's value into a common average and a layer-odd
difference,

$$
\mathbf{K}_\pm = \frac{\mathbf{K}_++\mathbf{K}_-}{2}
              \pm \frac{\mathbf{K}_+-\mathbf{K}_-}{2},
$$

resolves the shift into two physically distinct couplings:

- the **average** momentum $\tfrac12(\mathbf{K}_++\mathbf{K}_-)$ is a rigid boost
  shared by both layers and couples to the ordinary spatial gradient
  $\nabla\log\Psi$;
- the **difference** momentum $\tfrac12(\mathbf{K}_+-\mathbf{K}_-)$ multiplies the
  layer sign $\hat\sigma_z$ ($+1$ on the bottom layer, $-1$ on the top), so it
  couples to the **$\sigma_z$-weighted drift** — the ordinary spatial gradient
  with the layer sign attached to each electron. This layer-pseudospin coupling
  is what the "SOC" name refers to, and it is nonzero only because the two
  layers occupy *different* valleys.

The estimator keeps this correction in layer-basis form. For Cartesian
component $a$ and electron $i$, it constructs

$$
K_i^a = \eta_i\operatorname{diag}(K_+^a,K_-^a),
\qquad
Q = \frac{1}{2}\operatorname{diag}(|\mathbf{K}_+|^2,|\mathbf{K}_-|^2),
$$

and evaluates

$$
E_{\text{soc}}
= i\sum_{a,i}\frac{\nabla_i^a(K_i^a\Psi)}{\Psi}
  + \sum_i\frac{Q_i\Psi}{\Psi}.
$$

Both the SOC and moire-potential estimators use shared determinant algebra for
these layer matrices. For any estimator matrix $O_i$, its transformed orbitals
are

$$
\Phi^O_{dij}
= \sum_{\ell,m}\chi_{i\ell}(O_i)_{\ell m}\phi_{m,dij}.
$$

Replacing row $i$ of $\Phi^0_d$ by the corresponding row of $\Phi^O_d$ gives a
rank-one-updated matrix $M_i^{O,d}$ and the numerator
$O_i\Psi=\sum_d\det M_i^{O,d}$. The shared ratio kernel reuses
$(\Phi^0_d)^{-1}$ for the potential and $Q$ terms. For the drift term, the
shared derivative-ratio kernel computes $\nabla_i(O_i\Psi)/\Psi$ — not the
gradient of $(O_i\Psi)/\Psi$ — by linearizing the transformed orbitals once and
using Sherman–Morrison row-replacement inverses for all spatial coordinates.

### CI vs. FCI: filling inference and translation projection

The workflow infers the filling from the geometry,

$$
\text{filling} = \frac{N_e}{\det(S)\cdot A_\text{lattice}/A_\text{moire}},
$$

and selects the wavefunction phase mode automatically:

- **Integer filling → `singlephase` (CI).** The fixed Bloch labels ($\mathbf{k}$
  points that fold to the supercell $\Gamma$) are inferred from the filling and
  `supercell_matrix`. Do not set `wf.k_com` in this mode.
- **Non-integer filling → `multiphase` (FCI).** All primitive-cell $\mathbf{k}$
  points are dressed with trainable multiphase factors. Translation
  symmetrization (below) is controlled by `wf.k_com`, which defaults to
  $(0,0)$ — the $\Gamma$-point FCI sector — when left unset, and can be set to
  another sector from YAML or the CLI.

## Translation Symmetry

Periodic systems carry two translation symmetries: a supercell translation
associated with the twist momentum $\mathbf{k}_s$, and a primitive-cell
translation associated with the center-of-mass momentum $\mathbf{k}_p$. To resolve
the topological degeneracy of FCI states, the app builds $\mathbf{k}_p$-symmetric
wavefunctions,

$$
\Psi_{\mathbf{k}_p}(\mathbf{r}_1, \ldots, \mathbf{r}_{N_e}) = \sum_{\mathbf{l} \in \text{supercell}} e^{i\mathbf{k}_p \cdot \mathbf{l}} \, \Psi_{\text{Net}}(\mathbf{r}_1 - \mathbf{l}, \ldots, \mathbf{r}_{N_e} - \mathbf{l}),
$$

where the sum runs over primitive-cell copies $\mathbf{l}$ inside one supercell,
and the twist momentum $\mathbf{k}_s$ is fixed by an overall phase
$e^{i\mathbf{k}_s\cdot\sum_i\mathbf{r}_i}$ set via `system.twist`.

FCI translation symmetrization is controlled by a single wavefunction knob,
`wf.k_com`, in simulation-cell reciprocal fractional coordinates:

- **Default (unset / `null`): $(0,0)$.** When running through the moire
  workflow, leaving `wf.k_com` unset selects the $\Gamma$-point FCI sector, so
  translation projection is always on for fractional filling. (Integer-filling
  CI runs use `singlephase` mode and ignore `wf.k_com` entirely.)
- **`wf.k_com: [kx, ky]` selects another sector.** Set it from YAML (a
  `k_com: [1.0, 1.0]` entry under the top-level `wf:` mapping) or as a CLI
  override (`wf.k_com=[1.0,1.0]`) to build the symmetrized wavefunction for
  that center-of-mass momentum.

When `wf.k_com` is set, the same wavefunction evaluates translated copies across
the supercell and combines them with center-of-mass phases. No separate
translation-symmetry block or enable flag is used.

## Estimators

The moire workflow enables the following estimators (see
[training](train.md) and [evaluation](eval.md) for config flags):

- **Energy** — kinetic, Coulomb interaction, moire potential, SOC, and
  the summed total energy.

For the physics and derivations behind each estimator, see
<project:../../guide/estimators/index.md>.

## Recommended Hyperparameters

The train preset is production-oriented: 50,000 iterations with a gradient
clipping window of 20, KFAC with learning rate $3\times10^{-3}$ and damping
$3\times10^{-4}$, and a joint MCMC sampler tuned for moire units (20 steps
between iterations, initial move width 0.02). A single determinant is used
deliberately to demonstrate a compact representation of the correlated state. See
<project:../../guide/sampling.md> for walker count and mixing behavior, and the
[training reference](train.md) for the full resolved config.

## Further Reading

- **Configuration reference** — <project:train.md>, <project:eval.md>
- **Estimator physics** — <project:../../guide/estimators/index.md>
- **Running evaluations** — <project:../../guide/running-workflows.md>

```{toctree}
:hidden:

train.md
eval.md
```
