# Energy Corrections

The energy reported during training (`total_energy`) is complex-valued because the magnetic kinetic energy term is complex. The **electronic variational energy** $E_v$ is its real part — the imaginary component is a finite-sampling artifact whose expectation value vanishes. $E_v$ includes only the kinetic and electron-electron Coulomb terms evaluated on the sphere. It does **not** include the background charge contribution or finite-size corrections needed to compare with literature values.

To obtain the corrected energy $E_c$ per electron reported in our paper, apply the following post-processing steps:

## Background Charge

We add the neutralizing background contribution on the Haldane sphere (Eq. 7 of our paper; see also [Jain, *Composite Fermions*, Cambridge University Press, 2007]):

$$E_\text{bg} = -\kappa\hbar\omega_c \frac{N^2}{2\sqrt{Q}}$$

where $\kappa = (e^2/\epsilon\ell)/(\hbar\omega_c)$ is the Landau level mixing parameter (corresponds to `interaction_strength` in the code), $Q = \text{flux}/2$, and $N$ is the total electron count.

## Density Correction

We subtract the zero-point energy $N\omega_c/2$ and apply a density correction factor to reduce finite-size dependence (Eq. 8 of our paper; [Morf, Phys. Rev. B 33, 2221 (1986)](https://doi.org/10.1103/PhysRevB.33.2221)):

$$E_c = \sqrt{\frac{2Q\nu}{N}}\left(E_v + E_\text{bg} - \frac{N\omega_c}{2}\right)$$

where $\nu$ is the filling factor. The flux-filling relationship on the sphere is $2Q = N/\nu - \mathcal{S}$, where $\mathcal{S}$ is the topological shift (e.g., $\mathcal{S} = 3$ for the Laughlin $\nu = 1/3$ state). All energies are in units of $\hbar\omega_c\kappa$.

## Quasiparticle and Quasihole Excitations

On the Haldane sphere, quasiparticle (qp) and quasihole (qh) excitations are ground states at shifted flux values. For the $\nu = 1/3$ state:

$$2Q^{\text{qp/qh}} = 3(N-1) \mp 1$$

Their background charge contribution differs from the ground state:

$$E_\text{bg}^{\text{qp/qh}} = -\kappa\hbar\omega_c \frac{N^2 - q^2}{2\sqrt{Q^{\text{qp/qh}}}}$$

where $|q| = 1/3$ is the fractional excess charge. We apply the density correction to each term separately to get the corrected transport gap:

$$E_c^{\text{gap}} = \sqrt{\frac{2Q^{\text{qp}}\nu}{N}}\left(E_v^{\text{qp}} + E_\text{bg}^{\text{qp}} - \frac{N\omega_c}{2}\right) + \sqrt{\frac{2Q^{\text{qh}}\nu}{N}}\left(E_v^{\text{qh}} + E_\text{bg}^{\text{qh}} - \frac{N\omega_c}{2}\right) - 2E_c$$
