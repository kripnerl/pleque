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
