Naming convention used in PLEQUE
================================

Coordinates
-----------


Here presented naming convention is used to read/create input/output dict/xarray files.

* 2D
    * ``R`` - Radial cylindrical coordinates with zero on machine axis
    * ``Z`` - Vertical coordinate with zero on machine geometrical axis
* 1D
    * ``psi_n``  - normalized poloidal magnetic flux with zero on magnetic axis and one
      on the last closed flux surface

      :math:`\psi_\mathrm{N} = \frac{\psi -
      \psi_\text{ax}}{\psi_\text{LCFS} - \psi_\text{ax}}`
    * ``psi_1dprof`` - poloidal magnetic flux; this coordinate axis is used only if ``psi_n`` is
      not found on the input. Output files uses implicitly ``psi_n`` axis.

2D profiles
-----------
* Required on the input
    * ``psi`` - poloidal magnetic flux
* Calculated
    * ...

1D profiles
-----------

* Required on the input
    * ``pressure``
    * ``pprime``
    * ``F`` - :math:`F = R B_\phi`
* Calculated
    * ``pprime`` - :math:`p \partial_\psi`
    * ``FFprime`` - :math:`FF' = F \partial_\psi F`
    * ``Fprime`` - :math:`F' = \partial_\psi F`

Attributes
----------
* To be written.