# Geometry

Distance and displacement functions for open boundary conditions (molecules), periodic boundary conditions (solids), and spherical geometry (quantum Hall).

## Open boundary conditions

```{eval-rst}
.. autofunction:: jaqmc.geometry.obc.pair_displacements_within

.. autofunction:: jaqmc.geometry.obc.pair_displacements_between
```

## Periodic boundary conditions

```{eval-rst}
.. autoclass:: jaqmc.geometry.pbc.DistanceType
.. autoclass:: jaqmc.geometry.pbc.SymmetryType

.. autofunction:: jaqmc.geometry.pbc.build_distance_fn
.. autofunction:: jaqmc.geometry.pbc.wrap_positions
.. autofunction:: jaqmc.geometry.pbc.get_symmetry_lat
.. autofunction:: jaqmc.geometry.pbc.get_distance_function
.. autofunction:: jaqmc.geometry.pbc.make_pbc_gaussian_proposal
.. autofunction:: jaqmc.geometry.pbc.scaled_f
.. autofunction:: jaqmc.geometry.pbc.scaled_g
```

## Spherical geometry

```{eval-rst}
.. autofunction:: jaqmc.geometry.sphere.sphere_proposal
```
