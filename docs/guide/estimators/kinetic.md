# Kinetic energy

For configuration options, see the estimator sections of each system's config reference ([molecule](#molecule-estimators), [solid](#solid-estimators), [hall](#hall-estimators)).

The kinetic energy estimator computes the local kinetic energy from the wavefunction Laplacian. Two geometry-specific variants are provided: Euclidean (molecules and solids) and spherical (Haldane sphere / FQHE).

## Euclidean kinetic energy

The local kinetic energy for a single electron configuration is

$$
E_\text{kin} = -\frac{1}{2}\frac{\nabla^2 \psi}{\psi}
  = -\frac{1}{2}\left[
      \nabla^2 \log\psi + |\nabla \log\psi|^2
  \right]
$$

The second form follows from the chain rule and is what the code uses, since the neural network outputs $\log\psi$ directly.

Computing the Laplacian $\nabla^2 \log\psi = \sum_i \partial^2 \log\psi / \partial r_i^2$ over all $3N$ electron coordinates is the expensive part. The `mode` parameter selects the strategy.

### Laplacian modes

**`scan` and `fori_loop`** use reverse-mode AD. They linearize the gradient $\nabla \log\psi$ and extract the diagonal of the Hessian via Jacobian-vector products with unit vectors:

$$
\frac{\partial^2 \log\psi}{\partial r_i^2} = \mathbf{e}_i^\top \, H \, \mathbf{e}_i
$$

This avoids materializing the full $O(N^2)$ Hessian. The two modes differ only in how the loop over $i$ is executed:

| Mode | JAX primitive | Characteristics |
|------|---------------|-----------------|
| `scan` | `jax.lax.scan` | Materializes all iterations; higher memory, faster compilation |
| `fori_loop` | `jax.lax.fori_loop` | One iteration at a time; constant memory, slower compilation |

**`forward_laplacian`** uses JaQMC's :mod:`jaqmc.laplacian` transform and requires JAX >= 0.7.1. Instead of extracting the Hessian diagonal, it propagates Laplacian information alongside the function evaluation in a single forward pass. This avoids the $3N$ sequential JVPs entirely and can be significantly faster for large systems.

The default `mode` is version-dependent: `forward_laplacian` on JAX >= 0.7.1, and `scan` on older JAX versions.

Most users can stop at that mode choice. If you need to call the transform
directly in your own estimator, the extension guide shows the usual
coordinate-only closure pattern plus sparse and weighted seeding. It also
covers when direct use is enough, when
{func}`~jaqmc.laplacian.custom_laplacian` becomes worth adding, and where the
complex-number and primitive-handler contracts live. Continue to
<project:/extending/forward-laplacian/index.md>.

## Spherical kinetic energy

For the fractional quantum Hall (FQHE) workflow on a Haldane sphere with monopole
strength $Q$, the local kinetic energy is

$$
E_\text{kin} = \frac{|\Lambda|^2 \psi}{2R^2 \psi}.
$$

Here $\Lambda = \vec r\times(-i\nabla-A)$ is the kinetic angular momentum and
$R$ is the sphere radius (defaulting to $\sqrt{Q}$). The physics is
straightforward (see section 3.10.3 of *Composite Fermions* by Jain); the
difficulty is numerical. Two distinct singularities must be removed before the
derivatives can be evaluated stably:

1. **Coordinate singularity.** In the usual polar-coordinate representation,
   measuring how rapidly the wavefunction changes per unit physical distance
   requires dividing longitudinal derivatives by $\sin\theta$, which diverges
   at the poles.
2. **Gauge singularity.** Under a fixed global gauge, the wavefunction itself
   carries an arbitrary phase singularity at a pole.

Fixing the numerical stability therefore requires addressing both:
stereographic coordinates replace the singular polar coordinates, and a local
Wu--Yang gauge removes the phase singularity at the corresponding pole.

#### Regular coordinates and gauge

For each electron, the estimator uses the stereographic plane centered on the
nearer pole:

$$
\rho =
\begin{cases}
\tan(\theta/2), & \theta \leq \pi/2 \quad \text{(north chart)}, \\
\tan((\pi-\theta)/2), & \theta > \pi/2 \quad \text{(south chart)},
\end{cases}
\qquad
(x,y) = \rho(\cos\phi,\sin\phi).
$$

The selected pole is always $(x,y)=(0,0)$, and selecting the nearer pole keeps
$\rho\leq1$. The wavefunction is evaluated from the matching local spinor
before its kinetic derivatives are taken:

$$
(u_N,v_N) = \frac{(1,x-iy)}{\sqrt{1+\rho^2}},
\qquad
(u_S,v_S) = \frac{(x+iy,1)}{\sqrt{1+\rho^2}}.
$$

At the north pole, $(u_N,v_N)=(1,0)$. At the south pole,
$(u_S,v_S)=(0,1)$. These values do not depend on the arbitrary longitude.
Using coordinates alone would leave the gauge phase singular; using the local
gauge alone would leave the polar-coordinate factors singular.

#### Converting plane derivatives to sphere derivatives

The stereographic plane is only a coordinate map of the sphere. Distances in
the plane and distances on the unit sphere are related by

$$
ds^2 = \frac{4}{(1+\rho^2)^2}(dx^2+dy^2).
$$

It follows that the spherical second-derivative operator is the ordinary
plane Laplacian multiplied by

$$
g = \frac{(1+\rho^2)^2}{4}.
$$

Thus, $g_i$ is the derivative-conversion factor for electron $i$. It is
determined by the coordinate map, not fitted or approximated.

The local gauge also changes the phase derivative. Its correction for
electron $i$ is

$$
A_i = \frac{2s_iQ}{1+\rho_i^2}(-y_i,x_i),
\qquad
s_i =
\begin{cases}
-1, & \text{north chart}, \\
+1, & \text{south chart}.
\end{cases}
$$

$A_i$ is the monopole gauge potential in the selected plane. It ensures that
changing between the north and south representations does not change the
kinetic energy.

With $\ell=\log\psi$, the exact local kinetic energy is

$$
E_\text{kin}
  = -\frac{1}{2R^2}
    \sum_i g_i\left[
      \Delta_i\ell
      + \nabla_i\ell\cdot\nabla_i\ell
      - 2i\,A_i\cdot\nabla_i\ell
      - A_i\cdot A_i
    \right].
$$

Here $\nabla_i$ and $\Delta_i$ are ordinary gradient and Laplacian operations
in electron $i$'s plane. The factor $g_i$ converts them to the sphere, and
$A_i$ supplies the monopole-gauge correction.

Spherical kinetic energy supports the same `mode` choices as Euclidean kinetic
energy.

## See also

- [Angular momentum](angular-momentum.md) for the conserved $L$
  observables, which use the same charts and spinor evaluation.
- Configuration: [Molecule](#molecule-estimators), [Solid](#solid-estimators),
  Hall [training](#hall-train-estimators) and
  [evaluation](#hall-estimators)
- API: {class}`~jaqmc.estimator.kinetic.EuclideanKinetic`, {class}`~jaqmc.estimator.kinetic.SphericalKinetic`
- Laplacian modes: {class}`~jaqmc.estimator.kinetic.LaplacianMode`
