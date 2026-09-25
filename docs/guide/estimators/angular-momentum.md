# Angular momentum

JaQMC provides angular-momentum estimators only for the Haldane sphere.
[`SphericalAngularMomentum`](../../api-reference/estimators.md) reports local
$L_z$, $L_z^2$, and $L^2$ during training and evaluation. Those averages do
not give the weight in each total-$L$ sector;
[`SphericalAngularMomentumProjection`](../../api-reference/estimators.md)
reconstructs those weights during Hall evaluation.

## Local $L_z$ and $L^2$

The estimator applies one scalar SU(2) rotation per Cartesian axis to every
electron spinor,

$$
U_a(t) = e^{-it\sigma_a/2},
$$

where $\sigma_a$ are the Pauli matrices and $t$ is the dummy rotation angle
about axis $a$.

On the sphere, the wavefunction can be treated as a function of monopole spinors
$z_i=(u_i,v_i)$ rather than of the bare angles $(\theta_i,\phi_i)$. A spinor
encodes the electron's position on the sphere together with the gauge phase it
carries in the monopole field, so multiplying each spinor by $U_a(t)$
performs the complete rotation in one step: it moves the electron on the
sphere and simultaneously applies the gauge transformation required by the
monopole background. With $U_a(t)$ acting on every spinor, define
$$
g_a(t)=\log\psi\left(U_a(t)z_1,\ldots,U_a(t)z_N\right).
$$
The angular-momentum operators are the generators of these rotations: rotating
the arguments by $U_a(t)$ acts on the wavefunction as
$e^{-itL_a}$, so differentiating $g_a$ at $t=0$ gives
$g_a'=-iL_a\psi/\psi$, and differentiating once more yields
the second-order identities. Hence, at $t=0$,
$$
\frac{L_z\psi}{\psi}=ig_z',\qquad
\frac{L_z^2\psi}{\psi}
  =-\left((g_z')^2+g_z''\right),
$$
and
$$
\frac{L^2\psi}{\psi}
  =-\sum_{a\in\{x,y,z\}}\left((g_a')^2+g_a''\right).
$$
Differentiating just three scalar rotation angles (independent of the electron count) avoids coordinate singularities and is cheaper than differentiating
every electron coordinate separately.

## Sector projection

A mixture of total-$L$ sectors can share the same $\langle L_z\rangle$ and
$\langle L^2\rangle$. The projection estimator reconstructs the sector weight
$p_L$ and energy $E_L$ from rotational overlaps. For Haar-distributed
rotations $g\in SU(2)$, it measures

$$
K(g)=
\frac{\langle\psi|U(g)|\psi\rangle}{\langle\psi|\psi\rangle},
\qquad
K_H(g)=
\frac{\langle\psi|H U(g)|\psi\rangle}{\langle\psi|\psi\rangle}.
$$

$U(g)$ rotates every electron's monopole spinor. Integrating against the
character $\chi_L(g)$ of angular momentum $L$ gives the sector weight $p_L$
and the energy kernel $h_L$:

$$
p_L=(2L+1)\int dg\,\chi_L(g)\,K(g),
\qquad
h_L=(2L+1)\int dg\,\chi_L(g)\,K_H(g),
\qquad
E_L=h_L/p_L.
$$

The implementation estimates both integrals as a joint Monte Carlo average
over electron configurations and Haar rotations. Each walker evaluation draws
`n_rotations` new rotations from its estimator RNG; rotations are not reused
across walkers or evaluation steps. Additional walkers and additional
evaluation steps both increase the number of rotation samples.

A definite sector has one $p_L$ consistent with 1 and the others consistent
with 0. If the requested $L$ range covers the state's support, the weights
sum to one within sampling error; a shortfall means missing sectors. $E_L$ is
a ratio, so its uncertainty grows when $p_L$ is small.

Enable the estimator during Hall evaluation; see
{ref}`hall-angular-projection`.

## See also

- Configuration: Hall [training](#hall-train-estimators)
  and [evaluation](#hall-estimators)
- API: {class}`~jaqmc.estimator.angular_momentum.SphericalAngularMomentum`,
  {class}`~jaqmc.estimator.angular_momentum.SphericalAngularMomentumProjection`
