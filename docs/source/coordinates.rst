Coordinates
===========

The tokamak is a curvilinear device which can be described in a number of coordinate systems: the Cartesian coordinates :math:`(X,Y,Z)`, the cylindrical coordinates :math:`(R, Z, \phi)`, the toroidal coordinates :math:`(r, \theta, \phi)` and many more. PLEQUE supports all of the above and more, see the :doc:`Straight Field Lines <straight_field_lines_link>` example notebook.


Accepted coordinates types
--------------------------

**1D - coordinates**

+------------------------+-----------+------------------------------+
| Coordinate             | Code      | Note                         |
+========================+===========+==============================+
|:math:`\psi_\mathrm{N}` | ``psi_n`` | Default 1D coordinate        |
+------------------------+-----------+------------------------------+
|:math:`\psi`            | ``psi``   |                              |
+------------------------+-----------+------------------------------+
|:math:`\rho`            | ``rho``   | :math:`\rho = \sqrt{\psi_n}` |
+------------------------+-----------+------------------------------+

**2D - coordinates**

+------------------------+--------------+-------------------------------------------------+
| Coordinate             | Code         | Note                                            |
+========================+==============+=================================================+
|:math:`(R, Z)`          | ``R, Z``     | Default 2D coordinate                           |
+------------------------+--------------+-------------------------------------------------+
|:math:`(r, \theta)`     | ``r, theta`` | Polar coordinates with respect to magnetic axis |
+------------------------+--------------+-------------------------------------------------+

**3D - coordinates**

+------------------------+---------------+-------------------------------------------------+
| Coordinate             | Code          | Note                                            |
+========================+===============+=================================================+
|:math:`(R, Z, \phi)`    | ``R, Z, phi`` | Default 3D coordinate                           |
+------------------------+---------------+-------------------------------------------------+
|:math:`(X, Y, Z)`       | ``X, Y, Z``   |                                                 |
+------------------------+---------------+-------------------------------------------------+

Array shape convention
----------------------

PLEQUE distinguishes between scalar quantities, vector quantities, and the
topology of the coordinate input.

Scalar quantities preserve the spatial shape of the requested coordinates:

* paired one-dimensional coordinate arrays with ``grid=False`` return
  ``(n_elements,)``;
* one-dimensional ``R`` and ``Z`` axes with ``grid=True`` return
  ``(n_z, n_r)``;
* mesh-shaped ``R`` and ``Z`` arrays with ``grid=False`` are evaluated
  elementwise and preserve the input mesh shape, usually ``(n_z, n_r)``.

Vector quantities are component-first and then follow the scalar spatial
shape:

* point evaluations return ``(n_dim, n_elements)``;
* grid evaluations return ``(n_dim, n_z, n_r)``.

For magnetic-field vectors the component order is ``(R, Z, phi)``. For
``nabla_psi`` the component order is ``(dpsi/dR, dpsi/dZ)``.

Internally, some file formats and spline objects store rectangular data as
``(R, Z)``. Public evaluation methods transpose only these true grid results
at the API boundary so users consistently see ``(Z, R)`` spatial ordering.

Some geometry functions, such as effective flux-expansion coefficients, use a
surface normal calculated from neighbouring coordinate points. These functions
require ordered non-grid path coordinates; rectangular grids and mesh-shaped
point arrays do not define the required local normal.
