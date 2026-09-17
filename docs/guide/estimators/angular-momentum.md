# Angular momentum

JaQMC currently provides angular-momentum observables only for the Haldane
sphere, where the estimator reports the total $L_z$, $L_z^2$, and $L^2$.

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

## See also

- Configuration: Hall [training](#hall-train-estimators)
  and [evaluation](#hall-estimators)
- API: {class}`~jaqmc.estimator.angular_momentum.SphericalAngularMomentum`
