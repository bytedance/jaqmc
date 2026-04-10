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

**`forward_laplacian`** uses forward-mode AD via the [folx](https://github.com/microsoft/folx) library and requires JAX >= 0.7.1. Instead of extracting the Hessian diagonal, it propagates Laplacian information alongside the function evaluation in a single forward pass. This avoids the $3N$ sequential JVPs entirely and can be significantly faster for large systems.
The `sparsity_threshold` option can also be set to a positive value in `forward_laplacian` mode. This is [handled by folx](https://github.com/microsoft/folx#sparsity) to automatically detect sparsity during compilation. A typical threshold recommended by folx is `6`.

For {class}`~jaqmc.estimator.kinetic.EuclideanKinetic`, the default `mode` is version-dependent: `forward_laplacian` on JAX >= 0.7.1, and `scan` on older JAX versions.

## Spherical kinetic energy

For the quantum Hall (FQHE) workflow on a Haldane sphere with monopole strength $Q$, the kinetic energy uses the covariant angular momentum operator $\Lambda$:

$$
E_\text{kin} = \frac{|\Lambda|^2 \psi}{2R^2 \psi}
  = \frac{1}{2R^2}\left[
      -R^2 \frac{\nabla^2_S\psi}{\psi}
      + (Q\cot\theta)^2
      + 2iQ \frac{\cot\theta}{\sin\theta}
        \frac{\partial\log\psi}{\partial\phi}
  \right]
$$

where $\nabla^2_S = \frac{1}{\sin\theta}\partial_\theta(\sin\theta\,\partial_\theta) + \frac{1}{\sin^2\theta}\partial^2_\phi$ is the spherical Laplacian and $R$ is the sphere radius (defaults to $\sqrt{Q}$). The formulas follow section 3.10.3 of *Composite Fermions* (Jain).

In `scan` or `fori_loop` mode, the full Hessian of $\log\psi$ with respect to $(\theta, \phi)$ is computed. This also yields the angular momentum observables $L_z$ and $L^2$ as byproducts (reported in the training output as `Lz` and `L_square`). In `forward_laplacian` mode, the Hessian is not available, so $L^2$ is computed by applying the angular momentum operator $\hat{L}$ twice ($L^2 = \hat{\mathbf{L}} \cdot \hat{\mathbf{L}}$) in a separate pass.

## See also

- Configuration: [Molecule](#molecule-estimators), [Solid](#solid-estimators), [Hall](#hall-estimators)
- API: {class}`~jaqmc.estimator.kinetic.EuclideanKinetic`, {class}`~jaqmc.estimator.kinetic.SphericalKinetic`, {class}`~jaqmc.estimator.kinetic.LaplacianMode`
